"""Per-task model/provider/base_url pin on delegate_task (tasks=[...] and the legacy single-goal shape).

Covers the regression guard (no pin -> today's exact shared-route behaviour), a per-task pin only
changing that one child, three different pins in one batch each landing on their own child, the
model-only / provider-only asymmetry (#delegate-per-task-model), and schema advertisement.
"""

import json
import threading
import unittest
from unittest.mock import MagicMock, patch

from tools.delegate_tool import DELEGATE_TASK_SCHEMA, delegate_task
from tools.delegate_tool_config import _task_credentials
from tools.delegate_tool_tasks import _normalize_task_list


def _make_mock_parent(depth=0):
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "test-key"
    parent.provider = "openrouter"
    parent.requested_provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent.provider_data_collection = ""
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent.capabilities = None
    parent.acp_command = None
    parent.acp_args = []
    parent.request_overrides = {}
    parent.openrouter_min_coding_score = None
    parent.max_tokens = None
    parent._fallback_chain = None
    parent._oneshot_children_spawned = 0
    return parent


class _StubbedAgentCase(unittest.TestCase):
    """Patches run_agent.AIAgent so delegate_task() builds children without any real model calls."""

    def _run(self, **kwargs):
        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "done", "completed": True, "api_calls": 1,
            }
            MockAgent.return_value = mock_child
            result = json.loads(delegate_task(parent_agent=self.parent, **kwargs))
        return result, MockAgent

    def setUp(self):
        self.parent = _make_mock_parent(depth=0)


class TestNoRoutePinIsByteForByteUnchanged(_StubbedAgentCase):
    """The critical regression guard: absent model/provider/base_url, every child still gets the
    single shared delegation default -- the exact kwargs AIAgent got before this feature existed."""

    @patch("tools.delegate_tool._load_config", return_value={})
    def test_unpinned_batch_inherits_parent_route_on_every_child(self, _cfg):
        result, MockAgent = self._run(
            tasks=[{"goal": "first task here"}, {"goal": "second task here"}],
        )
        self.assertNotIn("error", result)
        self.assertEqual(MockAgent.call_count, 2)
        for call in MockAgent.call_args_list:
            kwargs = call.kwargs
            self.assertEqual(kwargs["provider"], self.parent.provider)
            self.assertEqual(kwargs["model"], self.parent.model)
            self.assertEqual(kwargs["base_url"], self.parent.base_url)

    @patch("tools.delegate_tool._load_config", return_value={"model": "configured/model", "provider": "openrouter"})
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_unpinned_task_reuses_the_batch_creds_object(self, mock_resolve, _cfg):
        """No model/provider/base_url on the task -> _task_credentials returns batch_creds UNCHANGED
        (same object), so _resolve_delegation_credentials is called exactly once for the whole batch."""
        mock_resolve.return_value = {
            "model": "configured/model", "provider": "openrouter", "base_url": "https://openrouter.ai/api/v1",
            "api_key": "k", "api_mode": "chat_completions", "request_overrides": None,
        }
        self._run(tasks=[{"goal": "unpinned task one"}, {"goal": "unpinned task two"}])
        mock_resolve.assert_called_once()


class TestPerTaskModelPin(_StubbedAgentCase):
    @patch("tools.delegate_tool._load_config", return_value={})
    def test_one_pinned_sibling_does_not_affect_the_other(self, _cfg):
        result, MockAgent = self._run(
            tasks=[
                {"goal": "unpinned sibling task"},
                {"goal": "pinned sibling task", "model": "anthropic/claude-opus-5"},
            ],
        )
        self.assertNotIn("error", result)
        unpinned_kwargs, pinned_kwargs = (c.kwargs for c in MockAgent.call_args_list)
        self.assertEqual(unpinned_kwargs["model"], self.parent.model)
        self.assertEqual(pinned_kwargs["model"], "anthropic/claude-opus-5")
        # Model-only pin: provider/base_url stay on the inherited parent route (see
        # _task_credentials -- a model-only pin changes WHAT runs, not WHERE).
        self.assertEqual(pinned_kwargs["provider"], self.parent.provider)
        self.assertEqual(pinned_kwargs["base_url"], self.parent.base_url)


class TestThreeTasksThreeModelsOneBatch(_StubbedAgentCase):
    @patch("tools.delegate_tool._load_config", return_value={})
    @patch("tools.delegate_tool._get_oneshot_max_children", return_value=0)
    def test_each_child_gets_its_own_pinned_model(self, _oneshot, _cfg):
        result, MockAgent = self._run(
            tasks=[
                {"goal": "task alpha work", "model": "model-a"},
                {"goal": "task beta work", "model": "model-b"},
                {"goal": "task gamma work", "model": "model-c"},
            ],
        )
        self.assertNotIn("error", result)
        self.assertEqual(MockAgent.call_count, 3)
        models = [c.kwargs["model"] for c in MockAgent.call_args_list]
        self.assertEqual(models, ["model-a", "model-b", "model-c"])
        # All three inherited the same (unpinned) provider/base_url.
        for call in MockAgent.call_args_list:
            self.assertEqual(call.kwargs["provider"], self.parent.provider)


class TestPerTaskProviderPin(_StubbedAgentCase):
    @patch("tools.delegate_tool._load_config", return_value={})
    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    def test_provider_given_without_model_routes_but_keeps_default_model(self, mock_resolve, _cfg):
        mock_resolve.return_value = {
            "provider": "nous", "model": None, "base_url": "https://inference-api.nousresearch.com/v1",
            "api_key": "nous-key", "api_mode": "chat_completions",
        }
        result, MockAgent = self._run(tasks=[{"goal": "provider-only pin task", "provider": "nous"}])
        self.assertNotIn("error", result)
        kwargs = MockAgent.call_args.kwargs
        self.assertEqual(kwargs["provider"], "nous")
        self.assertEqual(kwargs["base_url"], "https://inference-api.nousresearch.com/v1")
        # No model pinned -> falls back to the parent's model (resolve_runtime_provider returned none).
        self.assertEqual(kwargs["model"], self.parent.model)
        mock_resolve.assert_called_once_with(requested="nous", target_model=None)

    @patch("tools.delegate_tool._load_config", return_value={})
    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    def test_model_given_without_provider_keeps_parent_transport(self, mock_resolve, _cfg):
        result, MockAgent = self._run(tasks=[{"goal": "model-only pin task", "model": "only-the-model"}])
        self.assertNotIn("error", result)
        kwargs = MockAgent.call_args.kwargs
        self.assertEqual(kwargs["model"], "only-the-model")
        self.assertEqual(kwargs["provider"], self.parent.provider)
        self.assertEqual(kwargs["base_url"], self.parent.base_url)
        # A model-only pin never consults the provider resolver.
        mock_resolve.assert_not_called()

    @patch("tools.delegate_tool._load_config", return_value={})
    @patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=Exception("boom"))
    def test_unknown_provider_pin_fails_the_call_before_any_child_spawns(self, mock_resolve, _cfg):
        result, MockAgent = self._run(
            tasks=[{"goal": "task one is fine"}, {"goal": "task two bad pin", "provider": "nonexistent-provider"}],
        )
        self.assertIn("error", result)
        self.assertNotIn("results", result)
        MockAgent.assert_not_called()


class TestLegacySingleGoalShapeAcceptsRoutePin(_StubbedAgentCase):
    @patch("tools.delegate_tool._load_config", return_value={})
    def test_top_level_model_kwarg_pins_the_single_child(self, _cfg):
        result, MockAgent = self._run(goal="a single legacy-shape goal", model="legacy-pinned-model")
        self.assertNotIn("error", result)
        self.assertEqual(MockAgent.call_args.kwargs["model"], "legacy-pinned-model")


class TestTaskCredentialsHelper(unittest.TestCase):
    """Unit-level coverage of the merge/validation helper independent of child construction."""

    def setUp(self):
        self.parent = _make_mock_parent()

    def test_no_pin_returns_batch_creds_object_unchanged(self):
        batch_creds = {"model": "m", "provider": "p", "base_url": "b", "api_key": "k", "api_mode": "am"}
        creds, err = _task_credentials({"goal": "g"}, {}, batch_creds, self.parent)
        self.assertIsNone(err)
        self.assertIs(creds, batch_creds)

    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    def test_bad_provider_pin_surfaces_as_an_error_not_an_exception(self, mock_resolve):
        mock_resolve.side_effect = ValueError("Unknown provider 'nope'")
        creds, err = _task_credentials(
            {"goal": "g", "provider": "nope"}, {}, {"model": "m"}, self.parent,
        )
        self.assertIsNotNone(err)
        self.assertIn("Unknown provider", err)


class TestSchemaAdvertisesRouteKeys(unittest.TestCase):
    def test_task_item_schema_has_model_provider_base_url(self):
        task_props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"]
        for key in ("model", "provider", "base_url"):
            self.assertIn(key, task_props)
            self.assertEqual(task_props[key]["type"], "string")
        # 'goal' stays the only required key -- the new ones are all optional.
        self.assertEqual(DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["required"], ["goal"])


class TestTaskValidationRejectsBadRouteTypes(unittest.TestCase):
    def test_non_string_model_is_rejected_before_any_spawn(self):
        normalized, error = _normalize_task_list(
            None, None, [{"goal": "task with a bad model type", "model": 123}], None, "leaf", 3,
        )
        self.assertIsNone(normalized)
        self.assertIn("model", error)

    def test_empty_string_provider_is_rejected(self):
        normalized, error = _normalize_task_list(
            None, None, [{"goal": "task with an empty provider"}, {"goal": "x", "provider": "   "}], None, "leaf", 3,
        )
        self.assertIsNone(normalized)
        self.assertIn("provider", error)


if __name__ == "__main__":
    unittest.main()

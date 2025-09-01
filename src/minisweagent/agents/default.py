"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation."""

import os
import platform
import re
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass

from jinja2 import Template

from minisweagent import Environment, Model


@dataclass
class AgentConfig:
    # The default settings are the bare minimum to run the agent. Take a look at the config files for improved settings.
    system_template: str = "You are a helpful assistant that can do anything."
    instance_template: str = (
        "Your task: {{task}}. Please reply with a single shell command in triple backticks. "
        "To finish, the first line of the output of the shell command must be 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'."
    )
    timeout_template: str = (
        "The last command <command>{{action['action']}}</command> timed out and has been killed.\n"
        "The output of the command was:\n <output>\n{{output}}\n</output>\n"
        "Please try another command and make sure to avoid those requiring interactive input."
    )
    format_error_template: str = "Please always provide EXACTLY ONE action in triple backticks."
    action_observation_template: str = "Observation: {{output}}"
    summary_template: str = (
        "Please summarize the following conversation round between user and assistant. "
        "Focus on the key actions taken and their results. Keep it concise but informative.\n\n"
        "User: {{user_message}}\n"
        "Assistant: {{assistant_message}}\n\n"
        "Summary:"
    )
    # 总结功能配置
    enable_summary: bool = True
    summary_model_name: str = "gpt-4"
    summary_model_api_key: str = ""
    summary_frequency: int = 1  # 每几轮进行一次总结
    step_limit: int = 0
    cost_limit: float = 3.0


class NonTerminatingException(Exception):
    """Raised for conditions that can be handled by the agent."""


class FormatError(NonTerminatingException):
    """Raised when the LM's output is not in the expected format."""


class ExecutionTimeoutError(NonTerminatingException):
    """Raised when the action execution timed out."""


class TerminatingException(Exception):
    """Raised for conditions that terminate the agent."""


class Submitted(TerminatingException):
    """Raised when the LM declares that the agent has finished its task."""


class LimitsExceeded(TerminatingException):
    """Raised when the agent has reached its cost or step limit."""


class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: Callable = AgentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        
        # 初始化总结模型
        self.summary_model = None
        if self.config.enable_summary:
            self.summary_model = self._init_summary_model()

    def _init_summary_model(self) -> Model:
        """初始化总结模型"""
        from minisweagent.models import get_model
        
        # 使用更轻量的模型进行总结
        summary_config = {
            "model_name": self.config.summary_model_name,
            "model_kwargs": {}
        }
        
        # 如果设置了独立的API密钥
        if self.config.summary_model_api_key:
            summary_config["model_kwargs"]["api_key"] = self.config.summary_model_api_key
        elif os.getenv("SUMMARY_MODEL_API_KEY"):
            summary_config["model_kwargs"]["api_key"] = os.getenv("SUMMARY_MODEL_API_KEY")
            
        try:
            return get_model(self.config.summary_model_name, summary_config)
        except Exception as e:
            print(f"Warning: Failed to initialize summary model: {e}")
            return None

    def render_template(self, template: str, **kwargs) -> str:
        cs = asdict(self.config) | asdict(self.env.config) | asdict(self.model.config) | platform.uname()._asdict()
        return Template(template).render(**kwargs, **cs, **os.environ)

    def add_message(self, role: str, content: str, **kwargs):
        self.messages.append({"role": role, "content": content, **kwargs})

    def add_history_message(self, role: str, content: str, **kwargs):
        self.history_messages.append({"role": role, "content": content, **kwargs})

    def summarize_round(self, user_message: str, assistant_message: str) -> str:
        """使用独立的总结模型进行总结"""
        if not self.summary_model:
            return f"Round summary: User provided observation, Assistant suggested action"
        
        summary_prompt = self.render_template(
            self.config.summary_template,
            user_message=user_message,
            assistant_message=assistant_message
        )
        
        # 使用独立的总结模型
        summary_messages = [
            {"role": "system", "content": "You are a helpful summarizer. Provide concise summaries of conversation rounds."},
            {"role": "user", "content": summary_prompt}
        ]
        
        try:
            summary_response = self.summary_model.query(summary_messages)
            return summary_response.get("content", "Summary generation failed")
        except Exception as e:
            # 如果总结模型失败，返回简单总结
            print(f"Summary failed: {e}")
            return f"Round summary: User provided observation, Assistant suggested action"

    def run(self, task: str) -> tuple[str, str]:
        """Run step() until agent is finished. Return exit status & message"""
        self.messages = []
        self.history_messages = []
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template, task=task))
        self.add_history_message("system", self.render_template(self.config.system_template))
        self.add_history_message("user", self.render_template(self.config.instance_template, task=task))

        while True:
            try:
                self.step()
            except NonTerminatingException as e:
                self.add_message("user", str(e))
            except TerminatingException as e:
                self.add_message("user", str(e))
                return type(e).__name__, str(e)

    def step(self) -> dict:
        """Query the LM, execute the action, return the observation."""
        return self.get_observation(self.query())

    def query(self) -> dict:
        """Query the model and return the response."""
        if 0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost:
            raise LimitsExceeded()
        response = self.model.query(self.messages)
        self.add_message("assistant", **response)
        self.add_history_message("assistant", **response)
        return response

    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the observation."""
        output = self.execute_action(self.parse_action(response))
        observation = self.render_template(self.config.action_observation_template, output=output)
        
        # 获取上一个assistant消息（动作）
        assistant_message = self.messages[-1]["content"]
        
        # 根据频率决定是否进行总结
        if (self.config.enable_summary and 
            self.summary_model and 
            self.model.n_calls % self.config.summary_frequency == 0):
            
            # 总结这一轮
            summary = self.summarize_round(observation, assistant_message)
            
            # 替换assistant消息为总结
            self.messages[-1] = {"role": "system", "content": f"Round Summary of last round: {summary}"}
        
        self.add_history_message("user", observation)
        return output

    def parse_action(self, response: dict) -> dict:
        """Parse the action from the message. Returns the action."""
        actions = re.findall(r"```bash\n(.*?)\n```", response["content"], re.DOTALL)
        if len(actions) == 1:
            return {"action": actions[0].strip(), **response}
        raise FormatError(self.render_template(self.config.format_error_template, actions=actions))

    def execute_action(self, action: dict) -> dict:
        try:
            output = self.env.execute(action["action"])
        except subprocess.TimeoutExpired as e:
            output = e.output.decode("utf-8", errors="replace") if e.output else ""
            raise ExecutionTimeoutError(
                self.render_template(self.config.timeout_template, action=action, output=output)
            )
        except TimeoutError:
            raise ExecutionTimeoutError(self.render_template(self.config.timeout_template, action=action, output=""))
        self.has_finished(output)
        return output

    def has_finished(self, output: dict[str, str]):
        """Raises Submitted exception with final output if the agent has finished its task."""
        lines = output.get("output", "").lstrip().splitlines()
        if lines and lines[0].strip() in ["MINI_SWE_AGENT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]:
            raise Submitted("\n".join(lines[1:]))

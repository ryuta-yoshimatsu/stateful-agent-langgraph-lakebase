from typing import Any, Generator, Optional, Sequence, Union, Dict

import mlflow
from databricks_langchain import (
    ChatDatabricks,
    VectorSearchRetrieverTool,
    DatabricksFunctionClient,
    UCFunctionToolkit,
    set_uc_function_client,
)
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.tool_node import ToolNode
from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)
import os
import logging

mlflow.langchain.autolog()

################################################################
#### Lakebase Part
###############################################################

import uuid
import logging
from langgraph.checkpoint.postgres import PostgresSaver
from databricks.sdk import WorkspaceClient
import requests

def get_db_uri():

    w = WorkspaceClient(
        host=os.getenv("HOST_URL"),
        azure_tenant_id=os.getenv("AZURE_TENANT_ID"),
        azure_client_id=os.getenv("AZURE_CLIENT_ID"),
        azure_client_secret=os.getenv("AZURE_CLIENT_SECRET"),
        auth_type="azure-client-secret",
        )

    instance_name = "stateful-agent-backend"

    cred = w.database.generate_database_credential(
        request_id=str(uuid.uuid4()), 
        instance_names=[instance_name],
    )
    
    instance = w.database.get_database_instance(name=instance_name)

    DB_URI = (
        f"postgresql://{os.getenv("AZURE_CLIENT_ID")}:{cred.token}"
        f"@instance-75eabdf6-13f6-43a9-a9b8-d844c306d095.database.azuredatabricks.net:5432/"
        f"databricks_postgres?sslmode=require"
    )

    return DB_URI

checkpointer = PostgresSaver.from_conn_string(get_db_uri())

# some of this will need to be on the fly - only a one hour token??

################################################################

client = DatabricksFunctionClient()
set_uc_function_client(client)

############################################
# Define your LLM endpoint and system prompt
############################################
LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

system_prompt = """"""

###############################################################################
## Define tools for your agent, enabling it to retrieve data or take actions
## beyond text generation
## To create and see usage examples of more tools, see
## https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/agent-tool
###############################################################################
tools = []

# You can use UDFs in Unity Catalog as agent tools
uc_tool_names = []
uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names)
tools.extend(uc_toolkit.tools)

# # (Optional) Use Databricks vector search indexes as tools
# # See https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/unstructured-retrieval-tools
# # for details
#
# # TODO: Add vector search indexes as tools or delete this block
# vector_search_tools = [
#         VectorSearchRetrieverTool(
#         index_name="",
#         # filters="..."
#     )
# ]
# tools.extend(vector_search_tools)

#####################
## Define agent logic
#####################

def create_tool_calling_agent(
    model: LanguageModelLike,
    tools: Union[Sequence[BaseTool], ToolNode],
    system_prompt: Optional[str] = None,
    checkpointer: Optional[PostgresSaver] = None,
) -> CompiledGraph:
    model = model.bind_tools(tools)

    # Define the function that determines which node to go to
    def should_continue(state: ChatAgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # If there are function calls, continue. else, end
        if last_message.get("tool_calls"):
            return "continue"
        else:
            return "end"

    if system_prompt:
        preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": system_prompt}]
            + state["messages"]
        )
    else:
        preprocessor = RunnableLambda(lambda state: state["messages"])
    model_runnable = preprocessor | model

    def call_model(
        state: ChatAgentState,
        config: RunnableConfig,
    ):
        response = model_runnable.invoke(state, config)

        return {"messages": [response]}

    workflow = StateGraph(ChatAgentState)

    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("tools", ChatAgentToolNode(tools))

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")
    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)
    else:
        return workflow.compile()


class LangGraphChatAgent(ChatAgent):
    def __init__(self): #, agent: CompiledStateGraph):
        self.agent = None #agent

    def _make_config(
        self, 
        thread_id, 
        configurable_kwargs: Optional[dict] = None, 
        metadata_kwargs: Optional[dict] = None):
        if not configurable_kwargs:
            configurable_kwargs = {}
        if not metadata_kwargs:
            metadata_kwargs = {}
        configurable_dict = configurable_kwargs
        metadata_dict = metadata_kwargs
        configurable_dict.update({'thread_id': thread_id})
        config = {
            "configurable": configurable_dict,
            "metadata": metadata_dict
        }
        return config
    
    def handle_custom_inputs_as_config(self, custom_inputs: Optional[Dict[str,Any]] = None):
        if not custom_inputs:
            logging.warning('no custom inputs provided - will start a new thread_id')
            custom_inputs = dict()

        if "thread_id" not in custom_inputs:
            logging.warning('no thread_id provided, creating one')
            custom_inputs['thread_id'] = str(uuid.uuid4())
        else:
            logging.info(f'using thread_id {custom_inputs["thread_id"]}')

        thread_id = custom_inputs.pop("thread_id")
        metadata_dict = custom_inputs
        config = self._make_config(thread_id=thread_id,metadata_kwargs=metadata_dict)
        return config

    def get_history_of_thread(self, thread_id):
        db_uri = get_db_uri()
        with PostgresSaver.from_conn_string(db_uri) as checkpointer:
            agent = create_tool_calling_agent(llm, tools, system_prompt, checkpointer)

            history = list(
                agent.get_state_history(
                self.handle_custom_inputs_as_config(
                    custom_inputs= {
                    'thread_id' : thread_id,
                    })))

            if history:
                previous_messages = history[0].values['messages']
            else:
                previous_messages = []

        return previous_messages

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[Dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        # self.get_agent(llm, tools, system_prompt)
        db_uri = get_db_uri()
        with PostgresSaver.from_conn_string(db_uri) as checkpointer:

            agent = create_tool_calling_agent(llm, tools, system_prompt, checkpointer)
            
            request = {"messages": self._convert_messages_to_dict(messages)}
            print("custom inputs = ", custom_inputs)
            config = self.handle_custom_inputs_as_config(custom_inputs=custom_inputs)

            messages = []
            for event in agent.stream(request, config, stream_mode="updates"):
                for node_data in event.values():
                    messages.extend(
                        ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
                    )
            return ChatAgentResponse(messages=messages)

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[Dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        db_uri = get_db_uri()
        with PostgresSaver.from_conn_string(db_uri) as checkpointer:
            agent = create_tool_calling_agent(llm, tools, system_prompt, checkpointer)

            request = {"messages": self._convert_messages_to_dict(messages)}
            print("custom inputs = ", custom_inputs)
            config = self.handle_custom_inputs_as_config(custom_inputs=custom_inputs)


            for event in agent.stream(request, config, stream_mode="updates"):
                for node_data in event.values():
                    yield from (
                        ChatAgentChunk(**{"delta": msg}) for msg in node_data["messages"]
                    )

# Create the agent object, and specify it as the agent object to use when
# loading the agent back for inference via mlflow.models.set_model()
AGENT = LangGraphChatAgent() #agent)
mlflow.models.set_model(AGENT)

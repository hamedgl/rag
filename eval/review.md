# Eval drafts: review

Reply with e.g. `drop d3 d17`, `fix d22: <better question or answer>`, or `all good`.

52 items: {'lookup': 13, 'symptom': 10, 'exact': 8, 'code': 7, 'how-to': 8, 'decline': 6}


## d1 · lookup · langchain/frontend/integrations/openui
**Q:** Why should the OpenUI system prompt be generated at application startup instead of within a component?
**Expected:** It should be called at startup rather than inside a component to prevent it from being recomputed on every render.
**Evidence (2):** `Call this once at startup,
// not inside a component, to avoid recomputing it on every ren` · `not inside a React component; the prompt is several kilobytes and should be computed once`

## d2 · symptom · langchain/middleware/built-in
**Q:** My agent is running out of tokens and hitting the context limit during long sessions with repeated tool executions. How can I automatically prune older tool results to stay within limits?
**Expected:** You can use the context editing middleware (specifically ContextEditingMiddleware and ClearToolUsesEdit) to manage conversation context by clearing out older tool call results while preserving recent ones.
**Evidence (3):** `Manage conversation context by clearing older tool call outputs when token limits are reac` · `The middleware applies context editing strategies when token limits are reached. The most ` · `ClearToolUsesEdit(
trigger=100000,
keep=3,
)`

## d3 · exact · langchain/middleware/built-in
**Q:** What is the default number of retries configured for ToolRetryMiddleware?
**Expected:** 2
**Evidence (2):** `Parameter `max_retries` (type: number, default: 2):` · ``max_retries` - Number of retry attempts (default: 2)`

## d4 · code · langchain/messages
**Q:** What method on a chat model is used to pass in a list of messages and receive an AIMessage response?
**Expected:** invoke
**Evidence (4):** `response = model.invoke(messages)  # Returns AIMessage` · `response = model.invoke(conversation)
print(response)  # AIMessage("J'adore créer des appl` · `ai_msg = model.invoke(messages)` · `messages = [system_msg, human_msg]
response = model.invoke(messages)  # Returns AIMessage`

## d5 · lookup · langchain/frontend/message-queues
**Q:** How can you initialize a submission queue in an Angular component using a stream instance?
**Expected:** In Angular, you can set up the submission queue by calling `injectSubmissionQueue(this.stream)` with the stream instance.
**Evidence (1):** `queue = injectSubmissionQueue(this.stream);`

## d6 · how-to · langchain/frontend/integrations/copilotkit
**Q:** How can I configure the CopilotKit agent with custom middleware when deploying LangGraph?
**Expected:** Pass a list of your desired middleware functions (such as normalize_context, CopilotKitMiddleware(), and apply_structured_output_schema) to the `middleware` parameter when calling `create_agent`.
**Evidence (3):** `middleware=[
normalize_context,
CopilotKitMiddleware(),
apply_structured_output_schema,
],` · `Add it to the `middleware` list for create_agent or create_deep_agent.` · `agent = create_agent(
model="openai:gpt-5.5",
middleware=[
normalize_context,
CopilotKitMi`

## d7 · symptom · langchain/streaming
**Q:** When streaming responses from nested workflows, how can I identify which specific agent is generating each output token?
**Expected:** You can inspect the metadata yielded with the stream messages; specifically, check for the key 'lc_agent_name' within the metadata dictionary when streaming with subgraphs enabled.
**Evidence (5):** `token, metadata = chunk["data"]` · `if agent_name := metadata.get("lc_agent_name"):` · `Each message event includes metadata that identifies the source agent.` · `When `subgraphs` is enabled, each streaming event includes a **namespace** that identifies`

## d8 · exact · langchain/frontend/human-in-the-loop
**Q:** What is the label of the button shown when all pending action decisions are made?
**Expected:** Submit All Decisions
**Evidence (2):** `Submit All Decisions` · `{allDecided && (
<button
className="rounded bg-green-600 px-4 py-2 text-white"
onClick={()`

## d9 · code · langchain/tools
**Q:** What attribute of the tool runtime object allows you to inspect execution metadata such as run and thread IDs?
**Expected:** runtime.execution_info
**Evidence (3):** `Access thread ID, run ID, and retry state from within a tool via `runtime.execution_info`` · `Access execution identity (thread ID, run ID) via `runtime.execution_info`` · `| **Execution Info** | Identity and retry information for the current execution (thread ID`

## d10 · lookup · langchain/multi-agent/skills-sql-assistant
**Q:** Which tool is registered by default on the SkillMiddleware class in the SQL assistant implementation?
**Expected:** The `load_skill` tool is registered on the `SkillMiddleware` class.
**Evidence (4):** `tools = [load_skill]` · `class SkillMiddleware(AgentMiddleware):
"""Middleware that injects skill descriptions into` · `# Register the load_skill tool as a class variable
tools = [load_skill]` · `The `load_skill` tool is registered as a class variable, making it available to the agent.`

## d11 · how-to · langchain/context-engineering
**Q:** How can developers manage intermediate operations like guardrails or logging between an agent's main execution steps?
**Expected:** Developers can use middleware to intercept the agent lifecycle between steps. This allows them to update the context (such as modifying state or conversation history) or alter the execution flow by jumping to different steps in the lifecycle.
**Evidence (3):** `Control what happens **between** the core agent steps - intercepting data flow to implemen` · `[middleware](/oss/langchain/middleware) is the mechanism that makes context engineering pr` · `| **[Life-cycle Context](#life-cycle-context)** | What happens between model and tool call`

## d12 · symptom · langchain/context-engineering
**Q:** My tool function fails because it doesn't have access to the caller's database connection or credentials. How can the tool retrieve these execution parameters from the agent invocation?
**Expected:** The tool can read variables like credentials and database connections directly from `runtime.context` (such as `runtime.context.api_key` and `runtime.context.db_connection`), which are populated when calling `agent.invoke` with the `context` parameter.
**Evidence (5):** `user_id = runtime.context.user_id
api_key = runtime.context.api_key
db_connection = runtim` · `result = agent.invoke(
{"messages": [{"role": "user", "content": "Get my data"}]},
context` · `@dataclass
class Context:
user_id: str
api_key: str
db_connection: str

@tool
def fetch_us` · `# Read from Runtime Context: get API key and DB connection
user_id = runtime.context.user_`

## d13 · exact · langchain/frontend/integrations/ai-elements
**Q:** What state string is assigned to tool calls generated by the getToolCalls helper?
**Expected:** "input-available"
**Evidence (1):** `state: "input-available" as const,`

## d14 · code · langchain/mcp/tools
**Q:** Which property on a LangChain tool holds MCP provenance details?
**Expected:** tool.metadata
**Evidence (3):** `Each adapted tool may carry its MCP provenance under an `mcp` namespace on the LangChain t` · `each tool carries its MCP provenance under an `mcp` namespace on its metadata` · `MCP provenance is available on the tool's metadata under `metadata["mcp"]``

## d15 · lookup · langchain/messages
**Q:** Which environment variable can you configure to enable standard content blocks in message content for external applications?
**Expected:** You can set the LC_OUTPUT_VERSION environment variable to "v1".
**Evidence (3):** `To do this, you can set the `LC_OUTPUT_VERSION` environment variable to `v1`.` · `export LC_OUTPUT_VERSION=v1` · `Optional serialization of standard blocks into `content` via `LC_OUTPUT_VERSION=v1` or `ou`

## d16 · lookup · langgraph/streaming
**Q:** Which function should be used to retrieve the object responsible for emitting custom data during graph streaming?
**Expected:** You obtain the stream writer by calling get_stream_writer().
**Evidence (4):** `writer = get_stream_writer()` · `# Access the stream writer to send custom data
writer = get_stream_writer()` · `Use `get_stream_writer` to access the stream writer and emit custom data.` · `# Get the stream writer to send custom data
writer = get_stream_writer()`

## d17 · how-to · langgraph/choosing-apis
**Q:** How can you set up dynamic routing from a node using LangGraph's StateGraph?
**Expected:** You can define a conditional function that inspects the workflow state and returns the next destination, then attach it to a node using `workflow.add_conditional_edges(source_node, condition_function)`.
**Evidence (3):** `workflow.add_conditional_edges("step1", should_analyze)` · `def should_analyze(state):
return "analyze" if state["step1_result"]["needs_analysis"] els` · `LangGraph provides a way to do so by returning a Command object from node functions:
```py`

## d18 · symptom · langgraph/use-graph-api
**Q:** How can I avoid having to manually define the same retry and timeout policies on every single node call in my graph?
**Expected:** You can use `set_node_defaults` on the graph to configure parameters like retry_policy, timeout, cache_policy, or error_handler globally across all nodes at once, rather than passing them individually to each `add_node`.
**Evidence (5):** `Use `set_node_defaults` to set `retry_policy`, `timeout`, `cache_policy`, or `error_handle` · `Use `set_node_defaults` to set `retry_policy`, `timeout`, `cache_policy`, or `error_handle` · `Instead of repeating the same `retry_policy=`, `error_handler=`, `timeout=`, or `cache_pol` · `To apply the same `retry_policy`, `timeout`, or `error_handler` to every node in a graph w`

## d19 · exact · langgraph/use-functional-api
**Q:** What kinds of failures is the built-in RetryPolicy primarily configured to handle by default?
**Expected:** It is optimized to retry specific network errors.
**Evidence (3):** `The default RetryPolicy is optimized for retrying specific network errors.` · `retries on **any** exception except the following (and their subclasses):  
- `ValueError`` · `uses the `default_retry_on` function, which retries on any exception except for the follow`

## d20 · code · langgraph/use-graph-api
**Q:** What base class from typing_extensions is used to define the schema for runtime configuration in a LangGraph graph?
**Expected:** TypedDict
**Evidence (3):** `from typing_extensions import TypedDict` · `# 1. Specify config schema
class ContextSchema(TypedDict):` · `from typing_extensions import TypedDict

# 1. Specify config schema
class ContextSchema(Ty`

## d21 · lookup · langgraph/streaming
**Q:** Which stream mode outputs the complete graph state following every execution step?
**Expected:** The `values` stream mode returns the full state after each step.
**Evidence (3):** `| [values](#graph-state) | `ValuesStreamPart` | Full state after each step. |` · ``values` streams the **full value** of the state after each step of the graph.` · `Use this to stream the **full state** of the graph after each step.  
```python
for chunk `

## d22 · how-to · langgraph/interrupts
**Q:** How can I safely handle errors in a node that also uses interrupts without accidentally stopping the pause behavior?
**Expected:** To avoid catching the internal exception used by the interrupt mechanism, keep the interrupt call outside of try/except blocks or catch only specific exception types instead of broad exceptions.
**Evidence (4):** `Separate `interrupt` calls from error-prone code` · `Use specific exception types in try/except blocks` · `Do not wrap `interrupt` calls in bare try/except blocks` · `Separate `interrupt` calls from error-prone code
* ✅ Use specific exception types in try/e`

## d23 · symptom · langgraph/event-streaming
**Q:** My workflow execution paused because it needs manual user approval. How can I examine the data sent with that pause request using the event stream?
**Expected:** You can inspect the interrupt data by using the `stream.interrupts` projection, which is designed to examine human-in-the-loop interrupt payloads.
**Evidence (2):** `| `stream.interrupts` | Inspect human-in-the-loop interrupt payloads. |` · `When a graph pauses for human input, inspect `stream.interrupted` and `stream.interrupts``

## d24 · code · langgraph/studio
**Q:** What CLI flag should you provide to langgraph dev to work around Safari blocking localhost connections when connecting to Studio?
**Expected:** --tunnel
**Evidence (2):** `Safari blocks `localhost` connections to Studio. To work around this, run the above comman` · `Use the `--tunnel` flag with your command to create a secure tunnel, as Safari has limitat`

## d25 · lookup · langgraph/frontend/custom-stream-channels
**Q:** What return type should the init method declare when registering custom stream channels?
**Expected:** The init method should return a dictionary mapping channel identifier strings to StreamChannel objects (dict[str, StreamChannel]).
**Evidence (5):** `def init(self) -> dict[str, StreamChannel]:` · `def init(self) -> dict:
return {"custom": self.log}` · `def init(self) -> dict[str, StreamChannel]:
return {"redactionStats": self.redaction_stats` · `def init(self) -> dict:
return {"tool_activity": self.activity}`

## d26 · lookup · deepagents/code/mcp-tools
**Q:** What is the endpoint URL for the MCP server that provides LangChain API reference documentation?
**Expected:** The URL for the reference-langchain MCP server is https://reference.langchain.com/mcp.
**Evidence (1):** `| `reference-langchain` | `https://reference.langchain.com/mcp` | Canonical API reference:`

## d27 · how-to · deepagents/permissions
**Q:** How can I restrict an agent's access to specific files and directories?
**Expected:** You can restrict file access by supplying a list of declarative permission rules to the `permissions=` parameter, which the agent's built-in filesystem tools will enforce.
**Evidence (6):** `Control which files and directories an agent can read or write to using declarative permis` · `Pass a list of rules to `permissions=` and the agent's built-in filesystem tools respect t` · `This model lets you restrict agents to specific directories (for example, `/workspace/`), ` · `Control which files and directories an agent can read or write to using declarative permis`

## d28 · symptom · deepagents/frontend/todo-list
**Q:** Why is my TodoList component rendering completely blank even though it hasn't finished loading yet, and how do I fix it?
**Expected:** The component may be returning null when empty. To show an indicator while waiting for the agent to plan, check if `todos.length === 0 && isLoading` and render a loading container (e.g., displaying a spinner with 'Agent is creating a plan...'), while only returning `null` when `todos.length === 0 && !isLoading`.
**Evidence (4):** `if (todos.length === 0 && !isLoading) {
return null;
}` · `if (todos.length === 0 && isLoading) {
return (
<div className="rounded-lg border bg-white` · `if (todos.length === 0 && isLoading) {
return (
<div className="rounded-lg border bg-white` · `Show the todo list only when `todos.length > 0`. Before the agent creates its
plan, there'`

## d29 · exact · deepagents/data-analysis
**Q:** What destination file path does the data analysis agent write its sales analysis Python script to?
**Expected:** /root/analyze_sales.py
**Evidence (3):** `'file_path': '/root/analyze_sales.py'` · `Updated file /root/analyze_sales.py` · `file_path: /root/analyze_sales.py`

## d30 · code · deepagents/code/hooks
**Q:** Which field in the PreToolUse hook output is ignored when attempting to modify arguments passed to a tool?
**Expected:** updatedInput (or PreToolUse.updatedInput)
**Evidence (1):** ``PreToolUse.updatedInput` | Mutation ignored; `allow` or `ask` uses the ordinary [PreToolU`

## d31 · lookup · deepagents/code/remote-sandboxes
**Q:** Under which entry-point group should developers register their custom sandbox provider for Deep Agents Code?
**Expected:** Custom sandbox providers should be registered under the `deepagents_code.sandbox_providers` entry-point group.
**Evidence (2):** `implement a `SandboxProvider` subclass and register it under the `deepagents_code.sandbox_` · `[project.entry-points."deepagents_code.sandbox_providers"]
acme = "acme_sandbox.provider:A`

## d32 · how-to · deepagents/frontend/todo-list
**Q:** How can I calculate the completion percentage for a collection of task items in React?
**Expected:** Determine the number of tasks marked with a completed status by filtering the list, and then divide that count by the total task count multiplied by 100 (rounded to the nearest integer), returning 0 if the list is empty.
**Evidence (3):** `const completed = todos.filter((t) => t.status === "completed").length;` · `const percentage = todos.length
? Math.round((completed / todos.length) * 100)
: 0;` · `const completed = todos.filter((t) => t.status === "completed").length;
const percentage =`

## d33 · symptom · deepagents/code/configuration
**Q:** Why isn't my project's .env file able to disable or enable reading environment files using DEEPAGENTS_CODE_READ_PROJECT_DOTENV?
**Expected:** The setting to read dotenv files is resolved from managed configuration, system environment variables, and the user's config.toml before any project .env is evaluated. Consequently, a project .env cannot alter this setting for itself, and DEEPAGENTS_CODE_READ_PROJECT_DOTENV is explicitly restricted from being defined in dotenv files.
**Evidence (4):** `All of those are read before any project `.env` is applied, so a project `.env` cannot tur` · `The following keys cannot be set in either project or global dotenv files:` · ``DEEPAGENTS_CODE_READ_PROJECT_DOTENV`` · `The setting resolves from managed config, the environment variable, and the user `config.t`

## d34 · exact · deepagents/code/config-file
**Q:** What happens to the gateway URL in Deep Agents Code if a user overrides a gateway-provided API key with their own?
**Expected:** Deep Agents Code removes the gateway endpoint, directing requests straight to the provider instead.
**Evidence (2):** `if you replace a gateway-provisioned key with your own, Deep Agents Code also drops the ga` · `Deep Agents Code also drops the gateway endpoint, so your key goes to the provider directl`

## d35 · lookup · deepagents/code/config-file
**Q:** Which environment variable can be used to override the default trust policy for project extensions?
**Expected:** You can override the project extension policy by setting DEEPAGENTS_CODE_EXTENSIONS_TRUST.
**Evidence (4):** `Set `DEEPAGENTS_CODE_EXTENSIONS_TRUST` to override this value.` · `Set the default project policy to `ask`, `always`, or `never` with `[extensions].trust` or` · ``DEEPAGENTS_CODE_EXTENSIONS` overrides `enabled`, and `DEEPAGENTS_CODE_EXTENSIONS_TRUST` o` · `Set the default policy for project extensions in `<project>/.deepagents/extensions/`: `ask`

## d36 · lookup · python/integrations/vectorstores/google_bigquery_vector_search
**Q:** What types of metadata filters can be passed to similarity_search_by_vector in BigQueryVectorStore?
**Expected:** You can use both dictionary-based filters (e.g., passing a Python dict) and SQL-based string filters.
**Evidence (4):** `# Dictionary-based Filters
# This should only return "Banana" document.
docs = store.simil` · `# SQL-based Filters
# This should return "Banana", "Apples and oranges" and "Cars and airp` · `The vectorstore supports two methods for applying filters to metadata fields when performi` · `- SQL-based Filters
- Alternatively, you can provide a string representing an SQL WHERE cl`

## d37 · how-to · python/integrations/chat/anthropic
**Q:** How can I send an online PDF document to Claude for summarization using LangChain Anthropic?
**Expected:** You can create a HumanMessage containing a text prompt and a file block configured with the PDF's URL and the mime type 'application/pdf', then pass that message to ChatAnthropic's invoke method.
**Evidence (4):** `message = HumanMessage(
content=[
{"type": "text", "text": "Summarize this document."},
{
` · `"mime_type": "application/pdf",
},
]
)
response = model.invoke([message])` · `pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
pdf_da` · `content=[
{"type": "text", "text": "Summarize this document."},
{
"type": "file",
"url": "`

## d38 · symptom · python/integrations/tools/parallel_monitor
**Q:** Why am I unable to find a webhook signing secret for an individual monitor I created in Parallel?
**Expected:** Webhook signing secrets are not generated per-monitor; they are managed at the organization's webhook endpoint level within the Parallel dashboard.
**Evidence (1):** `The signing secret is configured at the org webhook-endpoint level in the Parallel dashboa`

## d39 · exact · python/integrations/middleware/azure_ai
**Q:** What is the default threshold value for severity when configuring image content moderation in Microsoft Foundry?
**Expected:** 4
**Evidence (2):** `Parameter `severity_threshold` (type: int, default: 4):` · `Parameter `severity_threshold` (type: int, default: 4):
Minimum severity score from `0` to`

## d40 · code · python/integrations/providers/microsoft
**Q:** What CLI command should be executed to test a Microsoft Foundry hosted agent project locally?
**Expected:** azd ai agent run
**Evidence (2):** ``azd ai agent run` to test it locally` · `To deploy either approach, use `azd ai agent init` to initialize a hosted-agent project, ``

## d41 · lookup · python/integrations/chat/ibm_watsonx
**Q:** What parameters are required to instantiate ChatWatsonx when using Cloud Pak for Data credentials?
**Expected:** To instantiate ChatWatsonx with Cloud Pak for Data credentials, you need to provide model_id, url, username, password, instance_id, version, project_id, and params.
**Evidence (1):** `chat = ChatWatsonx(
model_id="ibm/granite-34b-code-instruct",
url="PASTE YOUR URL HERE",
u`

## d42 · symptom · python/integrations/graphs/neo4j_cypher
**Q:** Why is my Cypher QA chain generating a natural language summary instead of giving me back the raw database records?
**Expected:** By default, the chain uses an LLM to synthesize an answer. To bypass the natural language response and output the raw database query results directly, set the parameter `return_direct=True` when creating the chain.
**Evidence (3):** `You can return direct results from the Cypher QA Chain using the `return_direct` parameter` · `return_direct=True,` · `The `return_direct` modifier specifies whether to return the direct results of the execute`

## d43 · exact · python/integrations/chat/bedrock
**Q:** Which Python package provides the ChatBedrockConverse model in LangChain?
**Expected:** langchain-aws
**Evidence (3):** `[`ChatBedrockConverse`](https://reference.langchain.com/python/langchain-aws/chat_models/b` · `| [`ChatBedrockConverse`](https://reference.langchain.com/python/langchain-aws/chat_models` · `from langchain_aws import ChatBedrockConverse`

## d44 · lookup · integrations/splitters/code_splitter
**Q:** Which Language enum value is passed to RecursiveCharacterTextSplitter to split PowerShell scripts?
**Expected:** You pass Language.POWERSHELL to RecursiveCharacterTextSplitter.from_language.
**Evidence (1):** `powershell_splitter = RecursiveCharacterTextSplitter.from_language(
language=Language.POWE`

## d45 · how-to · versioning
**Q:** How can I safely update between minor versions when using LangChain or LangGraph 1.0 LTS?
**Expected:** LTS versions adhere to semantic versioning (semver), which ensures that upgrading between minor versions can be done safely without breaking changes.
**Evidence (3):** `LTS releases follow semantic versioning (semver), allowing safe upgrades between minor ver` · `Users can upgrade between minor versions (e.g., 1.0 to 1.1) without breaking changes` · `Minor releases contain new features and improvements but do not include breaking changes.`

## d46 · symptom · contributing/documentation
**Q:** Why is my pull request blocked from merging after updating the docs?
**Expected:** Pull requests cannot be merged if automated CI/CD checks fail, which enforce the project's linting, formatting, and test standards.
**Evidence (2):** `All pull requests are automatically checked by CI/CD. The same linting and formatting stan` · `The same linting and formatting standards will be enforced, and PRs cannot be merged if th`

## d47 · decline · deepagents/context-engineering
**Q:** What is the exact maximum token threshold or hard limit enforced on AGENTS.md before the system truncates it when injecting it into the system prompt?
**Expected:** Not in docs
**Evidence (0):** 

## d48 · decline · python/integrations/vectorstores/milvus
**Q:** What is the peak memory consumption of the Milvus standalone server container when initializing an empty database instance?
**Expected:** Not in docs
**Evidence (0):** 

## d49 · decline · langchain/middleware/custom
**Q:** What is the exact latency reduction percentage observed when using prompt caching with this middleware on a 50k token context?
**Expected:** Not in docs
**Evidence (0):** 

## d50 · decline · python/integrations/vectorstores/pgvectorstore
**Q:** What is the query latency and memory usage difference when using HNSW versus IVFFlat indexes in PGVectorStore for 1024-dimension vectors at 1 million rows?
**Expected:** Not in docs
**Evidence (0):** 

## d51 · decline · deepagents/frontend/sandbox
**Q:** What is the maximum time-to-live (TTL) in minutes before an idle running coding session is purged from memory after a disconnect?
**Expected:** Not in docs
**Evidence (0):** 

## d52 · decline · langgraph/stores
**Q:** What is the maximum latency overhead added to graph execution when performing semantic search over 100,000 indexed store documents?
**Expected:** Not in docs
**Evidence (0):** 

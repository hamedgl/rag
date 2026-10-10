# Rare-token drafts: review

Reply with e.g. `drop r3`, `fix r2: <better question>`, or `all good`.


## r1 · error · python/integrations/chat/google_generative_ai
**Token:** `ClientError: 400 INVALID_ARGUMENT. Flex API is not supported for model: gemini-2.5-flash.`
**Q:** Why am I seeing "ClientError: 400 INVALID_ARGUMENT. Flex API is not supported for model: gemini-2.5-flash." instead of the request just falling back to standard pricing?
**Expected:** An unavailable service tier returns a 400 error rather than falling back to standard to prevent billing at an unexpected rate, as service tiers must be enabled for both the project and the model.
**Evidence (3):** `A tier has to be enabled for both the project and the model.` · `An unavailable tier returns a `400` instead of falling back to the standard tier, so the r` · `An unavailable tier returns a `400` instead of falling back to the standard tier, so the r`

## r2 · error · python/integrations/chat/anthropic
**Token:** `BadRequestError: Error code: 400 -`
**Q:** Why am I getting "BadRequestError: Error code: 400 -" when calling ChatAnthropic with claude-opus-5?
**Expected:** Newer Claude models such as claude-opus-5 reject non-default sampling parameters, so setting parameters like temperature, top_p, or top_k to non-default values causes a 400 error.
**Evidence (3):** `Newer Claude models reject non-default sampling parameters.` · `Setting `temperature`, `top_p`, or `top_k` to a non-default value on `claude-opus-5`, `cla` · `Non-default values rejected (400). Omit the parameters, or pass defaults only (for example`

## r3 · error · contributing/code
**Token:** `ValueError: If the model identifier is not recognized.`
**Q:** According to the documentation, when does the initialization function raise "ValueError: If the model identifier is not recognized."?
**Expected:** It is raised when the provided model identifier is not recognized.
**Evidence (2):** `ValueError: If the model identifier is not recognized.` · `Raises:
ValueError: If the model identifier is not recognized.`

## r4 · error · langgraph/errors/INVALID_GRAPH_NODE_RETURN_VALUE
**Token:** `InvalidUpdateError: Expected dict, got ['whoops']`
**Q:** Why am I receiving the error "InvalidUpdateError: Expected dict, got ['whoops']" when running my graph, and how do I fix it?
**Expected:** This error occurs because a node in your StateGraph returned a non-dict type (a list instead of a dict). To fix it, ensure your graph nodes return a dictionary containing one or more keys defined in your state.
**Evidence (3):** `A LangGraph `StateGraph`
received a non-dict return type from a node.` · `Nodes in your graph must return a dict containing one or more keys defined in your state.` · `If you have complex logic in your node, make sure all code paths return an appropriate dic`

## r5 · error · langchain/errors/MESSAGE_COERCION_FAILURE
**Token:** `ValueError: Message dict must contain 'role' and 'content' keys, got {'role': 'HumanMessage', 'random_field': 'random value'}`
**Q:** Why am I getting this error when invoking my ChatAnthropic model: "ValueError: Message dict must contain 'role' and 'content' keys, got {'role': 'HumanMessage', 'random_field': 'random value'}"?
**Expected:** The error occurs because message dictionaries passed to the model must contain both 'role' and 'content' keys, whereas the provided dictionary had 'role' and an invalid 'random_field' instead.
**Evidence (3):** `ValueError: Message dict must contain 'role' and 'content' keys, got {'role': 'HumanMessag` · `These include OpenAI style message objects (`{ role: "user", content: "Hello world!" }`), ` · `All inputs to chat models must be an array of LangChain message classes or a supported mes`

## r6 · env/config · deepagents/code/configuration
**Token:** `LANGGRAPH_DEFAULT_RECURSION_LIMIT`
**Q:** Where can I set LANGGRAPH_DEFAULT_RECURSION_LIMIT so that Deep Agents Code recognizes it?
**Expected:** You can set LANGGRAPH_DEFAULT_RECURSION_LIMIT in your shell environment or in the global ~/.deepagents/.env file. Deep Agents Code ignores this variable if set in a project-level .env file.
**Evidence (3):** `Set it in your shell or the global `~/.deepagents/.env`.` · `Deep Agents Code ignores it in a project `.env` because it bypasses the bounded `runtime.r` · `Set it in your shell or the global `~/.deepagents/.env`. Deep Agents Code ignores it in a `

## r7 · env/config · python/integrations/vectorstores/sap_hanavector
**Token:** `VEC_TEXT`
**Q:** What data type is used for the VEC_TEXT column in the LANGCHAIN_DEMO_NEW_TABLE table?
**Expected:** The data type for the VEC_TEXT column is NCLOB.
**Evidence (1):** `('VEC_TEXT', 'NCLOB')`

## r8 · env/config · langchain/multi-agent/handoffs-customer-support
**Token:** `WARRANTY_COLLECTOR_PROMPT`
**Q:** What tools are applied by the middleware when the system prompt is WARRANTY_COLLECTOR_PROMPT during Turn 1?
**Expected:** The middleware applies the `[record_warranty_status]` tool.
**Evidence (1):** `Middleware applies:
- System prompt: `WARRANTY_COLLECTOR_PROMPT`
- Tools: `[record_warrant`

## r9 · env/config · python/integrations/document_loaders/google_cloud_sql_mssql
**Token:** `DB_USER`
**Q:** What is the default sample value set for DB_USER in the Cloud SQL for SQL Server integration setup script?
**Expected:** The default value assigned to DB_USER is "sqlserver".
**Evidence (1):** `DB_USER = "sqlserver"  # @param {type:"string"}`

## r10 · env/config · deepagents/code/config-file
**Token:** `EXAMPLE_API_KEY`
**Q:** What setting in the config file is used to specify the environment variable name EXAMPLE_API_KEY for compatible providers?
**Expected:** The `api_key_env` setting is used to specify the environment variable name (such as `api_key_env = "EXAMPLE_API_KEY"`) under the provider configuration table.
**Evidence (1):** `api_key_env = "EXAMPLE_API_KEY"`

## r11 · import · python/integrations/chat/ollama
**Token:** `from PIL import Image`
**Q:** What package should I install via pip before running the multimodal ChatOllama code that contains `from PIL import Image`?
**Expected:** You need to install `pillow` by running `pip install pillow`.
**Evidence (2):** `pip install pillow` · ````bash
pip install pillow
```  
```python
import base64
from io import BytesIO

from IPyt`

## r12 · import · python/integrations/vectorstores/elasticsearch
**Token:** `from langchain_elasticsearch import BM25Strategy`
**Q:** What kind of retrieval does `from langchain_elasticsearch import BM25Strategy` enable when configuring an ElasticsearchStore?
**Expected:** Importing and using `BM25Strategy` enables full-text keyword search retrieval.
**Evidence (3):** `Finally, you can use full-text keyword search.` · `To use this, specify `BM25Strategy` in `ElasticsearchStore` constructor.` · `Finally, you can use full-text keyword search.  
To use this, specify `BM25Strategy` in `E`

## r13 · import · python/integrations/providers/upstage
**Token:** `from langchain_upstage import UpstageDocumentParseLoader`
**Q:** When using `from langchain_upstage import UpstageDocumentParseLoader`, which method should I call instead of load() for better memory efficiency?
**Expected:** You should use the `lazy_load()` method to load documents page by page for improved memory efficiency.
**Evidence (2):** `For improved memory efficiency, consider using the lazy_load method to load documents page` · `docs = layzer.load()  # or layzer.lazy_load()`

## r14 · import · contributing/implement-langchain
**Token:** `from langchain_tests.integration_tests import SandboxIntegrationTests`
**Q:** When writing standard tests with `from langchain_tests.integration_tests import SandboxIntegrationTests`, what fixture must you implement in the subclass?
**Expected:** You must implement a `sandbox` fixture that yields a clean `SandboxBackendProtocol` instance.
**Evidence (3):** `subclass it and provide a `sandbox` fixture that yields a clean `SandboxBackendProtocol` i` · `Subclass it and provide a `sandbox` fixture that yields a `SandboxBackendProtocol` instanc` · `class TestMySandboxStandard(SandboxIntegrationTests):
@pytest.fixture(scope="class")
def s`

## r15 · import · python/integrations/providers/ai21
**Token:** `from langchain_ai21 import AI21ContextualAnswers`
**Q:** Is the import `from langchain_ai21 import AI21ContextualAnswers` considered a deprecated feature?
**Expected:** Yes, it is listed under the deprecated features section for AI21 labs integrations.
**Evidence (2):** `AI21 labs integrations > Deprecated features > AI21 contextual answer` · `AI21 labs integrations > Deprecated features > AI21 contextual answer

```python
from lang`

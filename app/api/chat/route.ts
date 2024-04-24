import { kv } from '@vercel/kv'
import { OpenAIStream, StreamingTextResponse } from 'ai'
import { Configuration, OpenAIApi } from 'openai-edge'

import { auth } from '@/auth'
import { nanoid } from '@/lib/utils'

//new requires
import { NextRequest, NextResponse } from "next/server";
import { Message as VercelChatMessage} from "ai";

import { createClient } from "@supabase/supabase-js";

import { ChatOpenAI } from "langchain/chat_models/openai";
import { PromptTemplate } from "langchain/prompts";
import { SupabaseVectorStore } from "langchain/vectorstores/supabase";
import { Document } from "langchain/document";
import {
  RunnableSequence,
  RunnablePassthrough,
} from "langchain/schema/runnable";
import {
  BytesOutputParser,
  StringOutputParser,
} from "langchain/schema/output_parser";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";

//end new requires


export const runtime = 'edge'


//new code
type ConversationalRetrievalQAChainInput = {
  question: string;
  chat_history: VercelChatMessage[];
};

const combineDocumentsFn = (docs: Document[], separator = "\n\n") => {
  const serializedDocs = docs.map((doc) => doc.pageContent);
  return serializedDocs.join(separator);
};

const formatVercelMessages = (chatHistory: VercelChatMessage[]) => {
  const formattedDialogueTurns = chatHistory.map((message) => {
    if (message.role === "user") {
      return `Human: ${message.content}`;
    } else if (message.role === "assistant") {
      return `Assistant: ${message.content}`;
    } else {
      return `${message.role}: ${message.content}`;
    }
  });
  return formattedDialogueTurns.join("\n");
};

const CONDENSE_QUESTION_TEMPLATE = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;
const condenseQuestionPrompt = PromptTemplate.fromTemplate(
  CONDENSE_QUESTION_TEMPLATE,
);

const ANSWER_TEMPLATE = `You are a research assistant called Research Rover at the University of Zambia. You provide information on past projects carried out at the department of computer science and can suggest ideas for students to carry out. If the context does not have an answer suggest that it may not be in the database and give posible suggestions for the student. You are asked the following question:
Answer the question based only on the following context:
{context}

Question: {question}
`;
const answerPrompt = PromptTemplate.fromTemplate(ANSWER_TEMPLATE);

//end new code


const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY
})

const openai = new OpenAIApi(configuration)

export async function POST(req: Request) {
  const json = await req.json()
  const { messages, previewToken } = json
  const userId = (await auth())?.user.id

  if (!userId) {
    return new Response('Unauthorized', {
      status: 401
    })
  }

  if (previewToken) {
    configuration.apiKey = previewToken
  }

  // const res = await openai.createChatCompletion({
  //   model: 'gpt-3.5-turbo',
  //   messages,
  //   temperature: 0.7,
  //   stream: true
  // })

  // const stream = OpenAIStream(res, {
  //   async onCompletion(completion) {
  //     const title = json.messages[0].content.substring(0, 100)
  //     const id = json.id ?? nanoid()
  //     const createdAt = Date.now()
  //     const path = `/chat/${id}`
  //     const payload = {
  //       id,
  //       title,
  //       userId,
  //       createdAt,
  //       path,
  //       messages: [
  //         ...messages,
  //         {
  //           content: completion,
  //           role: 'assistant'
  //         }
  //       ]
  //     }
  //     await kv.hmset(`chat:${id}`, payload)
  //     await kv.zadd(`user:chat:${userId}`, {
  //       score: createdAt,
  //       member: `chat:${id}`
  //     })
  //   }
  // })

  // return new StreamingTextResponse(stream)

  //new code
  try {
  const previousMessages = messages.slice(0, -1);
    const currentMessageContent = messages[messages.length - 1].content;

    const model = new ChatOpenAI({
      modelName: "gpt-4",
    });

    const client = createClient(
      process.env.SUPABASE_URL!,
      process.env.SUPABASE_PRIVATE_KEY!,
    );
    const vectorstore = new SupabaseVectorStore(new OpenAIEmbeddings(), {
      client,
      tableName: "documents",
      queryName: "match_documents",
    });

    const retriever = vectorstore.asRetriever();

    /**
     * We use LangChain Expression Language to compose two chains.
     * To learn more, see the guide here:
     *
     * https://js.langchain.com/docs/guides/expression_language/cookbook
     */
    const standaloneQuestionChain = RunnableSequence.from([
      {
        question: (input: ConversationalRetrievalQAChainInput) =>
          input.question,
        chat_history: (input: ConversationalRetrievalQAChainInput) =>
          formatVercelMessages(input.chat_history),
      },
      condenseQuestionPrompt,
      model,
      new StringOutputParser(),
    ]);

    const answerChain = RunnableSequence.from([
      {
        context: retriever.pipe(combineDocumentsFn),
        question: new RunnablePassthrough(),
      },
      answerPrompt,
      model,
      new BytesOutputParser(),
    ]);

    const conversationalRetrievalQAChain =
      standaloneQuestionChain.pipe(answerChain);

    const stream = await conversationalRetrievalQAChain.stream({
      question: currentMessageContent,
      chat_history: previousMessages,
    });

    return new StreamingTextResponse(stream);
  } catch (e: any) {
    return NextResponse.json({ error: e.message }, { status: 500 });
  }
 //end new code
}

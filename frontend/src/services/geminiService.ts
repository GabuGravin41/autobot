/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { GoogleGenAI, Type } from "@google/genai";

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || '' });

export const generatePlan = async (prompt: string) => {
  const response = await ai.models.generateContent({
    model: "gemini-3.1-pro-preview",
    contents: `You are the Autobot AI Planner. Convert the user's request into a structured automation plan.
    
    User Request: ${prompt}
    
    Available Actions:
    - open_url(url: string)
    - adapter_call(adapter: string, action: string, args: object)
    - clipboard_set(text: string)
    - clipboard_get()
    - screenshot()
    - desktop_press(key: string)
    - wait(seconds: number)
    
    Available Adapters:
    - whatsapp_web: open_chat(phone, chat), send_message(text)
    - google_docs_web: open_new_doc(), open_by_url(url), type_text(text)
    - grok_web: ask_from_clipboard(), copy_response()
    - overleaf_web: replace_editor_text(text), compile(), download_pdf()
    
    Return a JSON object representing the plan.`,
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          name: { type: Type.STRING },
          description: { type: Type.STRING },
          steps: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                action: { type: Type.STRING },
                args: { type: Type.OBJECT },
                description: { type: Type.STRING }
              },
              required: ["action", "args", "description"]
            }
          }
        },
        required: ["name", "description", "steps"]
      }
    }
  });

  return JSON.parse(response.text || '{}');
};

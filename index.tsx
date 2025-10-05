import React, { useState, useRef, useEffect, useMemo } from 'react';
import ReactDOM from 'react-dom/client';
import { GoogleGenAI, Chat as GeminiChat, Type, Modality } from "@google/genai";
import type { LiveServerMessage, GroundingChunk } from "@google/genai";
import JSZip from 'jszip';

// =================================================================================
// --- TYPES ---
// =================================================================================

interface AITool {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
}

interface ChatMessage {
  role: 'user' | 'model';
  parts: string;
}

interface GeneratedImage {
  url: string;
  alt: string;
}

interface WebSearchResult {
    text: string;
    sources: GroundingChunk[];
}

interface TranscriptionEntry {
    speaker: 'user' | 'model';
    text: string;
    isFinal: boolean;
}

interface GeneratedFile {
  path: string;
  content: string;
}

type LiveSession = Awaited<ReturnType<typeof connectToLiveSession>>;
type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

interface SourceImage {
    base64: string;
    dataUrl: string;
    mimeType: string;
}

interface SpinnerProps {
    size?: 'sm' | 'md' | 'lg';
}
interface CodeBlockProps {
  code: string;
}
interface HeaderProps {
  activeTool: AITool;
  setActiveTool: (tool: AITool) => void;
}
interface FileExplorerProps {
  files: string[];
  activeFile: string;
  onSelectFile: (path: string) => void;
}
interface PreviewProps {
  files: GeneratedFile[];
}

// =================================================================================
// --- GEMINI AI SERVICE ---
// =================================================================================

if (!process.env.API_KEY) {
    console.warn("API_KEY environment variable not set. Using a placeholder.");
}
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || 'MISSING_API_KEY' });//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
export const ai = new GoogleGenAI({ apiKey: import.meta.env.VITE_GEMINI_API_KEY });//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// --- Chat Service ---
const initializeChatSession = (): GeminiChat => {
    return ai.chats.create({
        model: 'gemini-2.5-flash',
        config: {
            systemInstruction: 'You are a helpful and friendly AI assistant.',
        },
    });
};

const streamChat = async (chat: GeminiChat, message: string) => {
    return await chat.sendMessageStream({ message });
};

// --- Live Chat Service ---
const connectToLiveSession = (callbacks: {
    onopen: () => void;
    onmessage: (message: LiveServerMessage) => void;
    onerror: (e: any) => void;
    onclose: (e: any) => void;
}) => {
    return ai.live.connect({
        model: 'gemini-2.5-flash-native-audio-preview-09-2025',
        callbacks,
        config: {
            responseModalities: [Modality.AUDIO],
            speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Zephyr' } } },
            systemInstruction: 'You are a friendly and helpful AI assistant.',
            inputAudioTranscription: {},
            outputAudioTranscription: {},
        },
    });
};

function decode(base64: string) {
    const binaryString = atob(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes;
}

async function decodeAudioData(
    data: Uint8Array,
    ctx: AudioContext,
    sampleRate: number,
    numChannels: number,
): Promise<AudioBuffer> {
    const dataInt16 = new Int16Array(data.buffer);
    const frameCount = dataInt16.length / numChannels;
    const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

    for (let channel = 0; channel < numChannels; channel++) {
        const channelData = buffer.getChannelData(channel);
        for (let i = 0; i < frameCount; i++) {
            channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
        }
    }
    return buffer;
}

function encode(bytes: Uint8Array) {
    let binary = '';
    const len = bytes.byteLength;
    for (let i = 0; i < len; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
}

function createPcmBlob(data: Float32Array) {
    const l = data.length;
    const int16 = new Int16Array(l);
    for (let i = 0; i < l; i++) {
        int16[i] = data[i] * 32768;
    }
    return {
        data: encode(new Uint8Array(int16.buffer)),
        mimeType: 'audio/pcm;rate=16000',
    };
}


// --- Image Generation Service ---
const generateImages = async (prompt: string, aspectRatio: string): Promise<string[]> => {
    const response = await ai.models.generateImages({
        model: 'imagen-4.0-generate-001',
        prompt,
        config: {
            numberOfImages: 4,
            outputMimeType: 'image/jpeg',
            aspectRatio,
        },
    });

    return response.generatedImages.map(img => `data:image/jpeg;base64,${img.image.imageBytes}`);
};

// --- Image Editing Service ---
const editImage = async (base64ImageData: string, mimeType: string, prompt: string) => {
    const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash-image',
        contents: {
            parts: [
                { inlineData: { data: base64ImageData, mimeType } },
                { text: prompt },
            ],
        },
        config: {
            responseModalities: [Modality.IMAGE, Modality.TEXT],
        },
    });

    let imageUrl: string | null = null;
    let text: string | null = null;

    for (const part of response.candidates[0].content.parts) {
        if (part.text) {
            text = part.text;
        } else if (part.inlineData) {
            const base64ImageBytes: string = part.inlineData.data;
            imageUrl = `data:${part.inlineData.mimeType};base64,${base64ImageBytes}`;
        }
    }
    return { imageUrl, text };
};

// --- Video Generation Service ---
const generateVideo = async (prompt: string): Promise<string> => {
    let operation = await ai.models.generateVideos({
      model: 'veo-2.0-generate-001',
      prompt,
      config: {
        numberOfVideos: 1
      }
    });
    
    while (!operation.done) {
      await new Promise(resolve => setTimeout(resolve, 10000));
      operation = await ai.operations.getVideosOperation({operation: operation});
    }

    const downloadLink = operation.response?.generatedVideos?.[0]?.video?.uri;
    if (!downloadLink) {
        throw new Error("Video generation failed or did not return a download link.");
    }

    const response = await fetch(`${downloadLink}&key=${process.env.API_KEY || 'MISSING_API_KEY'}`);
    const videoBlob = await response.blob();
    return URL.createObjectURL(videoBlob);
};

// --- Web Search Service ---
const groundedSearch = async (prompt: string) => {
    const response = await ai.models.generateContent({
        model: "gemini-2.5-flash",
        contents: prompt,
        config: {
            tools: [{ googleSearch: {} }],
        },
    });

    return {
        text: response.text,
        sources: response.candidates?.[0]?.groundingMetadata?.groundingChunks ?? [],
    };
};

// --- JSON Generation Service ---
const generateRecipeJson = async (prompt: string) => {
    const response = await ai.models.generateContent({
        model: "gemini-2.5-flash",
        contents: `Generate a recipe for ${prompt}`,
        config: {
            responseMimeType: "application/json",
            responseSchema: {
                type: Type.OBJECT,
                properties: {
                    recipeName: { type: Type.STRING, description: 'The name of the recipe.' },
                    description: { type: Type.STRING, description: 'A short description of the dish.' },
                    prepTime: { type: Type.STRING, description: 'Preparation time, e.g., "20 minutes"' },
                    cookTime: { type: Type.STRING, description: 'Cooking time, e.g., "30 minutes"' },
                    servings: { type: Type.INTEGER, description: 'Number of servings.' },
                    ingredients: {
                        type: Type.ARRAY,
                        items: {
                            type: Type.OBJECT,
                            properties: {
                                amount: { type: Type.STRING, description: 'e.g., "1 cup" or "2 tbsp"' },
                                name: { type: Type.STRING, description: 'Name of the ingredient' },
                            },
                            required: ['amount', 'name'],
                        },
                    },
                    instructions: {
                        type: Type.ARRAY,
                        items: { type: Type.STRING, description: 'A single step in the recipe instructions.' },
                    },
                },
                required: ['recipeName', 'description', 'ingredients', 'instructions'],
            },
        },
    });

    return response.text;
};

// --- Build Service ---
const extractFilesFromJson = (jsonString: string): GeneratedFile[] => {
    try {
        const parsed = JSON.parse(jsonString);
        if (Array.isArray(parsed) && parsed.every(item => typeof item === 'object' && 'path' in item && 'content' in item)) {
            return parsed.map(item => ({ path: String(item.path), content: String(item.content) }));
        }
        throw new Error("JSON is not in the expected format of an array of files.");
    } catch (e: any) {
        throw new Error("The model did not return valid JSON for the file structure. " + e.message);
    }
};

const getBuildPrompt = (prompt: string, existingFiles: GeneratedFile[], type: 'web' | 'software') => {
    const fileList = existingFiles.map(f => `- ${f.path}`).join('\n');
    const projectType = type === 'web'
        ? 'a web application (HTML, CSS, JS)'
        : 'a software project (e.g., Python, JavaScript)';
        
    const instruction = existingFiles.length === 0
        ? `You are an expert software engineer. Your task is to generate the complete file structure and content for ${projectType} based on the user's prompt. The user wants: "${prompt}".`
        : `You are an expert software engineer. Your task is to update an existing project based on the user's request. The user's latest request is: "${prompt}".
        
The current files in the project are:
${fileList}

Analyze the user's request and provide the complete, updated set of all files for the project. If a file does not need to be changed, you MUST include it as-is in your response. Do not omit any files.`;

    return `${instruction}

You MUST respond with a single JSON array of objects, where each object has a "path" and a "content" key. Do not include any other text, explanations, or markdown formatting outside of the JSON array.

Example response format:
[
  {
    "path": "index.html",
    "content": "<!DOCTYPE html>..."
  },
  {
    "path": "style.css",
    "content": "body { ... }"
  }
]`;
};

const generateProject = async (prompt: string, existingFiles: GeneratedFile[], type: 'web' | 'software'): Promise<GeneratedFile[]> => {
    const fullPrompt = getBuildPrompt(prompt, existingFiles, type);

    const response = await ai.models.generateContent({
        model: "gemini-2.5-flash",
        contents: fullPrompt,
        config: {
            responseMimeType: "application/json",
            responseSchema: {
                type: Type.ARRAY,
                items: {
                    type: Type.OBJECT,
                    properties: {
                        path: { type: Type.STRING },
                        content: { type: Type.STRING },
                    },
                    required: ['path', 'content'],
                },
            },
        },
    });

    return extractFilesFromJson(response.text);
}


const generateWebApp = async (prompt: string, existingFiles: GeneratedFile[] = []): Promise<GeneratedFile[]> => {
    return generateProject(prompt, existingFiles, 'web');
};

const generateSoftwareProject = async (prompt: string, existingFiles: GeneratedFile[] = []): Promise<GeneratedFile[]> => {
    return generateProject(prompt, existingFiles, 'software');
};


// =================================================================================
// --- CONSTANTS ---
// =================================================================================
const AITools: AITool[] = [
  {
    id: 'chat',
    name: 'Chat',
    description: 'Engage in a free-flowing conversation with a powerful AI. Get answers, brainstorm ideas, and more.',
    icon: <span className="mr-2">💬</span>,
  },
  {
    id: 'live-chat',
    name: 'Live Chat',
    description: 'Experience a real-time voice conversation with Gemini. Speak and listen naturally.',
    icon: <span className="mr-2">🎙️</span>,
  },
  {
    id: 'image-generator',
    name: 'Image Gen',
    description: 'Bring your ideas to life. Generate stunning, high-quality images from text descriptions.',
    icon: <span className="mr-2">🖼️</span>,
  },
    {
    id: 'image-editor',
    name: 'Image Edit',
    description: 'Modify existing images with simple text prompts. Add objects, change styles, and more.',
    icon: <span className="mr-2">✏️</span>,
  },
  {
    id: 'video-generator',
    name: 'Video Gen',
    description: 'Create short video clips from text prompts. Animate your imagination.',
    icon: <span className="mr-2">🎬</span>,
  },
  {
    id: 'web-search',
    name: 'Web Search',
    description: 'Get up-to-date answers grounded in Google Search. Perfect for current events and recent topics.',
    icon: <span className="mr-2">🌐</span>,
  },
  {
    id: 'json-generator',
    name: 'JSON Output',
    description: 'Generate structured data in JSON format based on a schema and your prompt.',
    icon: <span className="mr-2">📦</span>,
  },
    {
    id: 'build',
    name: 'Build',
    description: 'Generate entire codebases for web or software projects. Describe your idea and watch it come to life.',
    icon: <span className="mr-2">🛠️</span>,
  },
];


// =================================================================================
// --- REUSABLE UI COMPONENTS ---
// =================================================================================

const Spinner: React.FC<SpinnerProps> = ({ size = 'md' }) => {
    const sizeClasses = {
        sm: 'h-5 w-5',
        md: 'h-8 w-8',
        lg: 'h-12 w-12',
    };

    return (
        <div className="flex justify-center items-center">
            <div
                className={`${sizeClasses[size]} animate-spin rounded-full border-4 border-slate-500 border-t-indigo-400`}
                role="status"
            >
                <span className="sr-only">Loading...</span>
            </div>
        </div>
    );
};

const CodeBlock: React.FC<CodeBlockProps> = ({ code }) => {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (copied) {
      const timer = setTimeout(() => setCopied(false), 2000);
      return () => clearTimeout(timer);
    }
  }, [copied]);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
  };

  return (
    <div className="bg-slate-900 rounded-lg overflow-hidden relative border border-slate-700">
      <button
        onClick={handleCopy}
        className="absolute top-2 right-2 bg-slate-700 hover:bg-slate-600 text-slate-300 px-3 py-1 text-xs font-bold rounded-md transition-colors"
      >
        {copied ? 'Copied!' : 'Copy'}
      </button>
      <pre className="p-4 text-sm text-slate-300 overflow-x-auto">
        <code>{code}</code>
      </pre>
    </div>
  );
};


const FileIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-2 text-slate-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
    </svg>
);

const FileExplorer: React.FC<FileExplorerProps> = ({ files, activeFile, onSelectFile }) => {
  return (
    <div className="bg-slate-800/50 p-3 rounded-lg border border-slate-700 h-full flex flex-col">
      <h3 className="text-sm font-bold text-slate-300 mb-2 px-1 flex-shrink-0">Files</h3>
      <ul className="space-y-1 overflow-y-auto">
        {files.map(path => (
          <li key={path}>
            <button
              onClick={() => onSelectFile(path)}
              className={`w-full text-left flex items-center px-2 py-1.5 text-sm rounded-md transition-colors ${
                activeFile === path
                  ? 'bg-indigo-600/50 text-white'
                  : 'text-slate-300 hover:bg-slate-700/50'
              }`}
            >
              <FileIcon />
              <span className="truncate">{path}</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
};


const Preview: React.FC<PreviewProps> = ({ files }) => {

  const srcDocContent = useMemo(() => {
    const htmlFile = files.find(f => f.path.toLowerCase().includes('index.html'));
    if (!htmlFile) {
      return '<!-- No index.html found to preview -->';
    }

    let processedHtml = htmlFile.content;

    const linkTagRegex = /<link\s+[^>]*?href="([^"]+)"[^>]*>/gi;
    processedHtml = processedHtml.replace(linkTagRegex, (match, href) => {
      if (!match.includes('rel="stylesheet"') && !href.endsWith('.css')) {
        return match;
      }
      const cssFile = files.find(f => f.path === href);
      if (cssFile) {
        return `<style>\n${cssFile.content}\n</style>`;
      }
      return `<!-- Linked file not found: ${href} -->`;
    });

    const scriptTagRegex = /<script\s+[^>]*?src="([^"]+)"[^>]*><\/script>/gi;
    processedHtml = processedHtml.replace(scriptTagRegex, (match, src) => {
      const jsFile = files.find(f => f.path === src);
      if (jsFile) {
        return `<script>\n${jsFile.content}\n</script>`;
      }
      return `<!-- Linked file not found: ${src} -->`;
    });

    return processedHtml;
  }, [files]);


  return (
    <div className="bg-slate-800/50 p-1 rounded-lg border border-slate-700 h-full flex flex-col">
       <div className="bg-slate-700/80 px-3 py-1.5 rounded-t-md flex-shrink-0">
           <h3 className="text-sm font-bold text-slate-300">Preview</h3>
       </div>
      <iframe
        srcDoc={srcDocContent}
        title="Web Preview"
        sandbox="allow-scripts allow-forms"
        className="w-full h-full bg-white rounded-b-md"
      />
    </div>
  );
};


// =================================================================================
// --- FEATURE COMPONENTS ---
// =================================================================================
const Header: React.FC<HeaderProps> = ({ activeTool, setActiveTool }) => {
  return (
    <header className="text-center">
      <h1 className="text-5xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-purple-400 mb-2">
        Gemini AI Studio
      </h1>
      <p className="text-slate-400 mb-8 max-w-2xl mx-auto">{activeTool.description}</p>
      <nav className="flex justify-center space-x-2 bg-slate-800 p-2 rounded-full border border-slate-700 shadow-md">
        {AITools.map((tool) => (
          <button
            key={tool.id}
            onClick={() => setActiveTool(tool)}
            className={`flex items-center justify-center px-4 py-2 rounded-full text-sm font-semibold transition-all duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-800 focus:ring-indigo-500 ${
              activeTool.id === tool.id
                ? 'bg-indigo-600 text-white shadow-lg'
                : 'text-slate-300 hover:bg-slate-700/50'
            }`}
          >
            {tool.icon}
            {tool.name}
          </button>
        ))}
      </nav>
    </header>
  );
};

const Chat: React.FC = () => {
  const [history, setHistory] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const chatSession = useRef<GeminiChat | null>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatSession.current = initializeChatSession();
  }, []);

  useEffect(() => {
    if (chatContainerRef.current) {
        chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [history]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading || !chatSession.current) return;
    
    const userMessage: ChatMessage = { role: 'user', parts: input };
    setHistory(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const stream = await streamChat(chatSession.current, input);
      let modelResponse = '';
      setHistory(prev => [...prev, { role: 'model', parts: '' }]);

      for await (const chunk of stream) {
        modelResponse += chunk.text;
        setHistory(prev => {
            const newHistory = [...prev];
            newHistory[newHistory.length - 1].parts = modelResponse;
            return newHistory;
        });
      }
    } catch (error) {
      console.error("Error sending message:", error);
      setHistory(prev => [...prev, { role: 'model', parts: "Sorry, I encountered an error. Please try again." }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="p-6 md:p-8 flex flex-col h-[70vh]">
      <div ref={chatContainerRef} className="flex-1 overflow-y-auto pr-4 -mr-4 space-y-6">
        {history.length === 0 && (
            <div className="text-center text-slate-400">Start the conversation by sending a message.</div>
        )}
        {history.map((msg, index) => (
          <div key={index} className={`flex items-start gap-4 ${msg.role === 'user' ? 'justify-end' : ''}`}>
            {msg.role === 'model' && (
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex-shrink-0"></div>
            )}
            <div className={`max-w-xl p-4 rounded-2xl ${msg.role === 'user' ? 'bg-indigo-600 text-white' : 'bg-slate-700 text-slate-200'}`}>
              <p className="whitespace-pre-wrap">{msg.parts}</p>
            </div>
          </div>
        ))}
         {isLoading && history[history.length -1]?.role === 'user' && (
             <div className="flex items-start gap-4">
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex-shrink-0"></div>
                <div className="max-w-xl p-4 rounded-2xl bg-slate-700 text-slate-200">
                    <div className="flex items-center space-x-2">
                        <span className="w-2 h-2 bg-slate-400 rounded-full animate-pulse-fast"></span>
                        <span className="w-2 h-2 bg-slate-400 rounded-full animate-pulse-fast animation-delay-200ms"></span>
                        <span className="w-2 h-2 bg-slate-400 rounded-full animate-pulse-fast animation-delay-400ms"></span>
                    </div>
                </div>
            </div>
        )}
      </div>
      <form onSubmit={handleSendMessage} className="mt-6 flex items-center gap-4">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          disabled={isLoading}
          className="flex-1 bg-slate-700 border border-slate-600 rounded-full px-6 py-3 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-all"
        />
        <button
          type="submit"
          disabled={isLoading || !input.trim()}
          className="bg-indigo-600 text-white rounded-full p-3 hover:bg-indigo-500 disabled:bg-slate-600 disabled:cursor-not-allowed transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-800 focus:ring-indigo-500"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
          </svg>
        </button>
      </form>
    </div>
  );
};


const ImageGenerator: React.FC = () => {
  const [prompt, setPrompt] = useState('');
  const [aspectRatio, setAspectRatio] = useState('1:1');
  const [images, setImages] = useState<GeneratedImage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const aspectRatios = ["1:1", "4:3", "3:4", "16:9", "9:16"];

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim() || isLoading) return;

    setIsLoading(true);
    setError(null);
    setImages([]);

    try {
      const generatedImageUrls = await generateImages(prompt, aspectRatio);
      const generatedImages: GeneratedImage[] = generatedImageUrls.map(url => ({ url, alt: prompt }));
      setImages(generatedImages);
    } catch (err) {
      console.error(err);
      setError("Failed to generate images. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="p-6 md:p-8">
      <form onSubmit={handleGenerate} className="flex flex-col md:flex-row gap-4 mb-8">
        <input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="e.g., A futuristic city skyline at sunset, cyberpunk style"
          className="flex-1 bg-slate-700 border border-slate-600 rounded-lg px-4 py-3 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-all"
        />
        <select
          value={aspectRatio}
          onChange={(e) => setAspectRatio(e.target.value)}
          className="bg-slate-700 border border-slate-600 rounded-lg px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
        >
          {aspectRatios.map(ar => <option key={ar} value={ar}>{ar}</option>)}
        </select>
        <button
          type="submit"
          disabled={isLoading || !prompt.trim()}
          className="bg-indigo-600 text-white font-bold rounded-lg px-6 py-3 hover:bg-indigo-500 disabled:bg-slate-600 disabled:cursor-not-allowed transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-800 focus:ring-indigo-500"
        >
          {isLoading ? 'Generating...' : 'Generate'}
        </button>
      </form>
      
      {error && <div className="text-center text-red-400">{error}</div>}

      {isLoading && (
        <div className="flex justify-center items-center h-64">
          <Spinner size="lg" />
        </div>
      )}

      {images.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {images.map((img, index) => (
            <div key={index} className="rounded-lg overflow-hidden border-2 border-slate-700 shadow-lg">
              <img src={img.url} alt={img.alt} className="w-full h-full object-cover" />
            </div>
          ))}
        </div>
      )}
    </div>
  );
};


const WebSearch: React.FC = () => {
  const [prompt, setPrompt] = useState('');
  const [result, setResult] = useState<WebSearchResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim() || isLoading) return;

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const searchResult = await groundedSearch(prompt);
      setResult(searchResult);
    } catch (err) {
      console.error(err);
      setError("Failed to perform search. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="p-6 md:p-8">
      <form onSubmit={handleSearch} className="flex flex-col md:flex-row gap-4 mb-8">
        <input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Ask about recent events or topics..."
          className="flex-1 bg-slate-700 border border-slate-600 rounded-lg px-4 py-3 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-all"
        />
        <button
          type="submit"
          disabled={isLoading || !prompt.trim()}
          className="bg-indigo-600 text-white font-bold rounded-lg px-6 py-3 hover:bg-indigo-500 disabled:bg-slate-600 disabled:cursor-not-allowed transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-800 focus:ring-indigo-500"
        >
          {isLoading ? 'Searching...' : 'Search'}
        </button>
      </form>

      {error && <div className="text-center text-red-400">{error}</div>}

      {isLoading && (
        <div className="flex justify-center items-center h-64">
          <Spinner size="lg" />
        </div>
      )}

      {result && (
        <div className="space-y-6">
          <div className="bg-slate-700/50 p-6 rounded-lg border border-slate-600">
            <h2 className="text-xl font-bold mb-3 text-indigo-400">Response</h2>
            <p className="text-slate-300 whitespace-pre-wrap">{result.text}</p>
          </div>

          {result.sources.length > 0 && (
            <div className="bg-slate-700/50 p-6 rounded-lg border border-slate-600">
              <h3 className="text-lg font-bold mb-3 text-indigo-400">Sources</h3>
              <ul className="space-y-2">
                {result.sources.map((source, index) => (
                  <li key={index} className="flex items-center">
                    <span className="text-indigo-400 mr-2">&#10148;</span>
                    <a
                      href={source.web?.uri}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sky-400 hover:text-sky-300 hover:underline truncate"
                    >
                      {source.web?.title || source.web?.uri}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};


const JsonGenerator: React.FC = () => {
  const [prompt, setPrompt] = useState('');
  const [jsonOutput, setJsonOutput] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim() || isLoading) return;

    setIsLoading(true);
    setError(null);
    setJsonOutput(null);

    try {
      const generatedJson = await generateRecipeJson(prompt);
      try {
        const parsed = JSON.parse(generatedJson);
        setJsonOutput(JSON.stringify(parsed, null, 2));
      } catch {
        setJsonOutput(generatedJson); 
      }
    } catch (err) {
      console.error(err);
      setError("Failed to generate JSON. Please check your prompt and try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="p-6 md:p-8">
      <div className="mb-4 bg-indigo-900/50 border border-indigo-700 text-indigo-300 p-4 rounded-lg text-sm">
        <p><strong>Note:</strong> This tool generates a recipe in a specific JSON format. Try a prompt like "a spicy chicken pasta dish" or "a vegan chocolate cake".</p>
      </div>

      <form onSubmit={handleGenerate} className="flex flex-col md:flex-row gap-4 mb-8">
        <input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Describe the recipe you want..."
          className="flex-1 bg-slate-700 border border-slate-600 rounded-lg px-4 py-3 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-all"
        />
        <button
          type="submit"
          disabled={isLoading || !prompt.trim()}
          className="bg-indigo-600 text-white font-bold rounded-lg px-6 py-3 hover:bg-indigo-500 disabled:bg-slate-600 disabled:cursor-not-allowed transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-800 focus:ring-indigo-500"
        >
          {isLoading ? 'Generating...' : 'Generate JSON'}
        </button>
      </form>

      {error && <div className="text-center text-red-400 mb-4">{error}</div>}

      {isLoading && (
        <div className="flex justify-center items-center h-64">
          <Spinner size="lg" />
        </div>
      )}

      {jsonOutput && (
        <div>
            <h2 className="text-xl font-bold mb-3 text-indigo-400">Generated Recipe JSON</h2>
            <CodeBlock code={jsonOutput} />
        </div>
      )}
    </div>
  );
};


const LiveChat: React.FC = () => {
    const [status, setStatus] = useState<ConnectionStatus>('disconnected');
    const [transcription, setTranscription] = useState<TranscriptionEntry[]>([]);
    const [isModelSpeaking, setIsModelSpeaking] = useState(false);

    const sessionPromise = useRef<Promise<LiveSession> | null>(null);
    const sessionRef = useRef<LiveSession | null>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const audioContextRefs = useRef<{ input: AudioContext | null, output: AudioContext | null, processor: ScriptProcessorNode | null }>({
        input: null,
        output: null,
        processor: null
    });
    const playbackQueue = useRef<{ buffer: AudioBuffer, source: AudioBufferSourceNode }[]>([]);
    const nextStartTime = useRef(0);

    const cleanup = () => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }
        if (audioContextRefs.current.input) {
            audioContextRefs.current.input.close();
            audioContextRefs.current.input = null;
        }
        if (audioContextRefs.current.output) {
             playbackQueue.current.forEach(({ source }) => source.stop());
             playbackQueue.current = [];
            audioContextRefs.current.output.close();
            audioContextRefs.current.output = null;
        }
         if (audioContextRefs.current.processor) {
            audioContextRefs.current.processor.disconnect();
            audioContextRefs.current.processor = null;
        }
    };
    
    useEffect(() => {
        return () => {
            if (sessionRef.current) {
                sessionRef.current.close();
                sessionRef.current = null;
            }
            cleanup();
        };
    }, []);

    const processTranscription = (message: LiveServerMessage) => {
        let speaker: 'user' | 'model' | null = null;
        let text = '';
        if (message.serverContent?.inputTranscription) {
            speaker = 'user';
            text = message.serverContent.inputTranscription.text;
        } else if (message.serverContent?.outputTranscription) {
            speaker = 'model';
            text = message.serverContent.outputTranscription.text;
        }

        if (speaker && text) {
            setTranscription(prev => {
                const newTranscription = [...prev];
                const lastEntry = newTranscription[newTranscription.length - 1];

                if (lastEntry && lastEntry.speaker === speaker && !lastEntry.isFinal) {
                    lastEntry.text += text;
                } else {
                    newTranscription.push({ speaker, text, isFinal: false });
                }
                return newTranscription;
            });
        }
        
        if (message.serverContent?.turnComplete) {
             setTranscription(prev => prev.map(t => ({ ...t, isFinal: true })));
        }
    };
    
    const handleStartSession = async () => {
        if (status === 'connected' || status === 'connecting') return;
        setStatus('connecting');
        setTranscription([]);

        try {
            streamRef.current = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioContextRefs.current.input = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
            audioContextRefs.current.output = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
            nextStartTime.current = 0;

            sessionPromise.current = connectToLiveSession({
                onopen: () => {
                    setStatus('connected');
                    const source = audioContextRefs.current.input!.createMediaStreamSource(streamRef.current!);
                    const processor = audioContextRefs.current.input!.createScriptProcessor(4096, 1, 1);
                    processor.onaudioprocess = (e) => {
                        const inputData = e.inputBuffer.getChannelData(0);
                        const pcmBlob = createPcmBlob(inputData);
                        sessionPromise.current?.then(session => session.sendRealtimeInput({ media: pcmBlob }));
                    };
                    source.connect(processor);
                    processor.connect(audioContextRefs.current.input!.destination);
                    audioContextRefs.current.processor = processor;
                },
                onmessage: async (message) => {
                    processTranscription(message);
                    const audioData = message.serverContent?.modelTurn?.parts[0]?.inlineData.data;
                    if (audioData) {
                        setIsModelSpeaking(true);
                        const outputContext = audioContextRefs.current.output!;
                        const audioBuffer = await decodeAudioData(decode(audioData), outputContext, 24000, 1);
                        const source = outputContext.createBufferSource();
                        source.buffer = audioBuffer;
                        source.connect(outputContext.destination);

                        const startTime = Math.max(nextStartTime.current, outputContext.currentTime);
                        source.start(startTime);
                        nextStartTime.current = startTime + audioBuffer.duration;

                        const queueEntry = { buffer: audioBuffer, source };
                        playbackQueue.current.push(queueEntry);
                        source.onended = () => {
                            playbackQueue.current = playbackQueue.current.filter(item => item !== queueEntry);
                            if (playbackQueue.current.length === 0) {
                                setIsModelSpeaking(false);
                            }
                        };
                    }
                },
                onerror: (e) => {
                    console.error('Session error:', e);
                    setStatus('error');
                    cleanup();
                },
                onclose: () => {
                    setStatus('disconnected');
                    cleanup();
                },
            });
            sessionRef.current = await sessionPromise.current;

        } catch (err) {
            console.error('Failed to start session:', err);
            setStatus('error');
            cleanup();
        }
    };
    
    const handleStopSession = () => {
        if (sessionRef.current) {
            sessionRef.current.close();
            sessionRef.current = null;
        }
        cleanup();
        setStatus('disconnected');
    };

    const StatusIndicator = () => {
        const statusConfig = {
            disconnected: { color: 'bg-slate-500', text: 'Disconnected' },
            connecting: { color: 'bg-yellow-500 animate-pulse', text: 'Connecting...' },
            connected: { color: 'bg-green-500', text: 'Connected' },
            error: { color: 'bg-red-500', text: 'Error' },
        };
        const { color, text } = statusConfig[status];
        
        return (
             <div className="flex items-center space-x-3 mb-4">
                <div className={`w-4 h-4 rounded-full ${color} transition-colors`}></div>
                <span className="text-slate-300">{text}</span>
            </div>
        )
    }

    return (
        <div className="p-6 md:p-8 flex flex-col h-[70vh]">
            <div className="flex-1 flex flex-col justify-center items-center text-center">
                {transcription.length === 0 ? (
                    <>
                        <div className="relative mb-6">
                             {isModelSpeaking && <div className="absolute inset-0 bg-indigo-500/30 rounded-full animate-pulse"></div>}
                            <div className={`relative w-24 h-24 rounded-full flex items-center justify-center transition-colors ${status === 'connected' ? 'bg-indigo-600' : 'bg-slate-700'}`}>
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-white" viewBox="0 0 20 20" fill="currentColor">
                                     <path d="M7 4a3 3 0 016 0v6a3 3 0 11-6 0V4z" />
                                     <path d="M5.5 4.5a2.5 2.5 0 015 0v6a2.5 2.5 0 01-5 0V4.5z" />
                                     <path d="M10 15a6 6 0 006-6h-1.5a4.5 4.5 0 01-9 0H4a6 6 0 006 6z" />
                                     <path d="M10 12a1 1 0 011 1v2a1 1 0 11-2 0v-2a1 1 0 011-1z" />
                                </svg>
                            </div>
                        </div>
                        <StatusIndicator />
                        <p className="text-slate-400 max-w-sm">
                            Press "Start Session" to begin a real-time voice conversation with Gemini.
                        </p>
                    </>
                ) : (
                    <div className="w-full h-full overflow-y-auto pr-4 -mr-4 space-y-6">
                        {transcription.map((entry, index) => (
                             <div key={index} className={`flex items-start gap-4 ${entry.speaker === 'user' ? 'justify-end' : ''}`}>
                                 {entry.speaker === 'model' && (
                                     <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex-shrink-0"></div>
                                 )}
                                <div className={`max-w-xl p-4 rounded-2xl ${entry.speaker === 'user' ? 'bg-indigo-600 text-white' : 'bg-slate-700 text-slate-200'} ${!entry.isFinal ? 'opacity-70' : ''}`}>
                                    <p>{entry.text}</p>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
             <div className="mt-6 flex justify-center items-center">
                {status === 'connected' ? (
                     <button onClick={handleStopSession} className="bg-red-600 text-white font-bold rounded-full px-8 py-4 hover:bg-red-500 transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-800 focus:ring-red-500">
                        Stop Session
                    </button>
                ) : (
                    <button onClick={handleStartSession} disabled={status === 'connecting'} className="bg-indigo-600 text-white font-bold rounded-full px-8 py-4 hover:bg-indigo-500 disabled:bg-slate-600 disabled:cursor-not-allowed transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-800 focus:ring-indigo-500">
                        {status === 'connecting' ? 'Connecting...' : 'Start Session'}
                    </button>
                )}
            </div>
        </div>
    );
};

const ImageEditor: React.FC = () => {
    const [sourceImage, setSourceImage] = useState<SourceImage | null>(null);
    const [prompt, setPrompt] = useState('');
    const [resultImage, setResultImage] = useState<string | null>(null);
    const [resultText, setResultText] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (loadEvent) => {
                const dataUrl = loadEvent.target?.result as string;
                const base64 = dataUrl.split(',')[1];
                setSourceImage({ base64, dataUrl, mimeType: file.type });
                setResultImage(null);
                setResultText(null);
            };
            reader.readAsDataURL(file);
        }
    };

    const handleGenerate = async () => {
        if (!sourceImage || !prompt.trim() || isLoading) return;

        setIsLoading(true);
        setError(null);
        setResultImage(null);
        setResultText(null);

        try {
            const { imageUrl, text } = await editImage(sourceImage.base64, sourceImage.mimeType, prompt);
            if (imageUrl) {
                setResultImage(imageUrl);
            }
            if (text) {
                setResultText(text);
            }
            if(!imageUrl && !text){
                 setError("The model didn't return an image or text. Try a different prompt.");
            }
        } catch (err) {
            console.error(err);
            setError("Failed to edit the image. Please try again.");
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="p-6 md:p-8">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div className="flex flex-col gap-6">
                    <div>
                        <label className="block text-sm font-medium text-slate-300 mb-2">1. Upload your image</label>
                        <div
                            className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-slate-600 border-dashed rounded-md cursor-pointer hover:border-indigo-500 transition-colors"
                            onClick={() => fileInputRef.current?.click()}
                        >
                            <div className="space-y-1 text-center">
                                {sourceImage ? (
                                    <img src={sourceImage.dataUrl} alt="Source preview" className="mx-auto h-40 w-auto rounded-md" />
                                ) : (
                                    <>
                                        <svg className="mx-auto h-12 w-12 text-slate-400" stroke="currentColor" fill="none" viewBox="0 0 48 48" aria-hidden="true">
                                            <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
                                        </svg>
                                        <div className="flex text-sm text-slate-400">
                                            <p className="pl-1">Click to upload an image</p>
                                        </div>
                                    </>
                                )}
                            </div>
                        </div>
                        <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={handleFileChange} />
                    </div>

                    <div>
                         <label htmlFor="prompt" className="block text-sm font-medium text-slate-300 mb-2">2. Describe your edit</label>
                         <textarea
                            id="prompt"
                            rows={3}
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                            placeholder="e.g., Add a futuristic helmet to the person"
                            className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-3 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-all"
                            disabled={!sourceImage}
                        />
                    </div>
                     <button
                        onClick={handleGenerate}
                        disabled={isLoading || !prompt.trim() || !sourceImage}
                        className="w-full bg-indigo-600 text-white font-bold rounded-lg px-6 py-4 text-lg hover:bg-indigo-500 disabled:bg-slate-600 disabled:cursor-not-allowed transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-800 focus:ring-indigo-500"
                    >
                        {isLoading ? 'Generating...' : 'Generate Edit'}
                    </button>
                </div>

                <div className="flex flex-col gap-4">
                     <h3 className="text-lg font-semibold text-slate-200">Result</h3>
                     <div className="flex-grow bg-slate-900/50 rounded-lg border border-slate-700 flex justify-center items-center min-h-[300px] p-4">
                        {isLoading ? (
                            <Spinner size="lg" />
                        ) : error ? (
                            <p className="text-red-400 text-center">{error}</p>
                        ) : resultImage ? (
                            <img src={resultImage} alt="Edited result" className="max-h-full max-w-full object-contain rounded-md" />
                        ) : (
                            <p className="text-slate-500 text-center">Your edited image will appear here</p>
                        )}
                    </div>
                    {resultText && (
                        <div className="bg-slate-700/50 p-4 rounded-lg border border-slate-600">
                           <p className="text-slate-300 italic">{resultText}</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};


const VideoGenerator: React.FC = () => {
    const [prompt, setPrompt] = useState('');
    const [videoUrl, setVideoUrl] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    
    const loadingMessages = [
        "Warming up the video engine...",
        "Rendering pixels into motion...",
        "Compositing the digital scenes...",
        "This can take a few minutes, please wait...",
        "Almost there, adding the final touches...",
    ];
    const [loadingMessage, setLoadingMessage] = useState(loadingMessages[0]);

    useEffect(() => {
        let interval: number;
        if (isLoading) {
            interval = window.setInterval(() => {
                setLoadingMessage(prev => {
                    const currentIndex = loadingMessages.indexOf(prev);
                    const nextIndex = (currentIndex + 1) % loadingMessages.length;
                    return loadingMessages[nextIndex];
                });
            }, 3000);
        }
        return () => clearInterval(interval);
    }, [isLoading]);

    const handleGenerate = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!prompt.trim() || isLoading) return;

        setIsLoading(true);
        setError(null);
        setVideoUrl(null);
        setLoadingMessage(loadingMessages[0]);

        try {
            const url = await generateVideo(prompt);
            setVideoUrl(url);
        } catch (err) {
            console.error(err);
            setError("Failed to generate video. Please try again.");
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="p-6 md:p-8">
            <form onSubmit={handleGenerate} className="flex flex-col md:flex-row gap-4 mb-8">
                <input
                    type="text"
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="e.g., A cinematic shot of a car driving on a rainy night"
                    className="flex-1 bg-slate-700 border border-slate-600 rounded-lg px-4 py-3 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-all"
                />
                <button
                    type="submit"
                    disabled={isLoading || !prompt.trim()}
                    className="bg-indigo-600 text-white font-bold rounded-lg px-6 py-3 hover:bg-indigo-500 disabled:bg-slate-600 disabled:cursor-not-allowed transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-800 focus:ring-indigo-500"
                >
                    {isLoading ? 'Generating...' : 'Generate Video'}
                </button>
            </form>

            {error && <div className="text-center text-red-400">{error}</div>}

            {isLoading && (
                <div className="flex flex-col justify-center items-center h-64">
                    <Spinner size="lg" />
                    <p className="mt-4 text-slate-400">{loadingMessage}</p>
                </div>
            )}

            {videoUrl && (
                <div className="mt-8 flex justify-center">
                    <div className="w-full max-w-2xl rounded-lg overflow-hidden border-2 border-slate-700 shadow-lg bg-black">
                        <video src={videoUrl} controls autoPlay loop className="w-full h-full object-contain" />
                    </div>
                </div>
            )}
        </div>
    );
};


const Build: React.FC = () => {
  const [prompt, setPrompt] = useState('');
  const [buildType, setBuildType] = useState<'web' | 'software'>('web');
  const [files, setFiles] = useState<GeneratedFile[]>([]);
  const [activeFile, setActiveFile] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<string[]>([]);
  const historyContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (historyContainerRef.current) {
      historyContainerRef.current.scrollTop = historyContainerRef.current.scrollHeight;
    }
  }, [history]);
  
  const handleBuild = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim() || isLoading) return;

    const currentPrompt = prompt;
    setIsLoading(true);
    setError(null);
    setPrompt('');
    setHistory(prev => [...prev, currentPrompt]);

    try {
      const buildFunction = buildType === 'web' ? generateWebApp : generateSoftwareProject;
      const generatedFiles = await buildFunction(currentPrompt, files);
      
      if (generatedFiles.length > 0) {
        setFiles(generatedFiles);
        const newActiveFile = generatedFiles.find(f => f.path === activeFile)
            ? activeFile
            : generatedFiles.find(f => f.path.toLowerCase().includes('index.html'))?.path || generatedFiles[0].path;
        setActiveFile(newActiveFile);
      } else {
        setError("The AI didn't return any files. Try refining your prompt.");
      }
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Failed to build the application. Please try again.");
      setHistory(prev => prev.slice(0, -1)); 
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleStartOver = () => {
    setFiles([]);
    setPrompt('');
    setActiveFile('');
    setError(null);
    setHistory([]);
    setBuildType('web');
  }

  const handleDownload = async () => {
    if (files.length === 0) return;

    const zip = new JSZip();
    files.forEach(file => {
        zip.file(file.path, file.content);
    });

    try {
        const content = await zip.generateAsync({ type: 'blob' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(content);
        link.download = 'ai-studio-project.zip';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    } catch (e) {
        console.error("Failed to generate zip file", e);
        setError("Could not generate ZIP file for download.");
    }
  };

  const activeFileContent = useMemo(() => {
    return files.find(f => f.path === activeFile)?.content || '';
  }, [files, activeFile]);

  const filePaths = useMemo(() => files.map(f => f.path), [files]);

  return (
    <div className="p-4 md:p-6 h-[80vh] flex flex-col">
      <div className="flex-shrink-0 mb-4">
        <div className="flex justify-center mb-4">
            <div className="bg-slate-700 p-1 rounded-lg flex space-x-1">
                <button
                    onClick={() => setBuildType('web')}
                    className={`px-4 py-1.5 text-sm font-semibold rounded-md transition-colors ${buildType === 'web' ? 'bg-indigo-600 text-white' : 'text-slate-300 hover:bg-slate-600/50'}`}
                >
                    Web
                </button>
                <button
                    onClick={() => setBuildType('software')}
                    className={`px-4 py-1.5 text-sm font-semibold rounded-md transition-colors ${buildType === 'software' ? 'bg-indigo-600 text-white' : 'text-slate-300 hover:bg-slate-600/50'}`}
                >
                    Software
                </button>
            </div>
        </div>
        <form onSubmit={handleBuild} className="flex flex-col md:flex-row gap-4">
            <div className="flex-1 flex bg-slate-700 border border-slate-600 rounded-lg focus-within:ring-2 focus-within:ring-indigo-500 transition-all">
                <input
                  type="text"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder={
                    files.length === 0 
                      ? (buildType === 'web' ? "e.g., A simple portfolio website..." : "e.g., A python script for a CLI calculator")
                      : (buildType === 'web' ? "e.g., Now, add a dark mode toggle" : "e.g., Now add a function for subtraction")
                  }
                  className="w-full bg-transparent px-4 py-3 text-white placeholder-slate-400 focus:outline-none"
                />
            </div>
            <button
              type="submit"
              disabled={isLoading || !prompt.trim()}
              className="bg-indigo-600 text-white font-bold rounded-lg px-6 py-3 hover:bg-indigo-500 disabled:bg-slate-600 disabled:cursor-not-allowed transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-800 focus:ring-indigo-500"
            >
              {isLoading ? 'Building...' : (files.length === 0 ? 'Build' : 'Update')}
            </button>
            {files.length > 0 && (
                <>
                <button
                  type="button"
                  onClick={handleDownload}
                  title="Download Project as ZIP"
                  className="bg-purple-600 text-white font-bold rounded-lg px-4 py-3 hover:bg-purple-500 transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-800 focus:ring-purple-500"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                    </svg>
                </button>
                 <button
                  type="button"
                  onClick={handleStartOver}
                  className="bg-slate-600 text-white font-bold rounded-lg px-6 py-3 hover:bg-slate-500 transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-800 focus:ring-slate-500"
                >
                  Start Over
                </button>
                </>
            )}
        </form>
      </div>
      
      {error && <div className="text-center text-red-400 mb-4 p-3 bg-red-900/50 border border-red-700 rounded-lg">{error}</div>}

      {isLoading && (
        <div className="flex-grow flex justify-center items-center">
          <div className="text-center">
            <Spinner size="lg" />
            <p className="mt-4 text-slate-400">Thinking... this may take a moment.</p>
          </div>
        </div>
      )}

      {!isLoading && files.length > 0 && (
         <div className="flex-grow grid grid-cols-1 lg:grid-cols-12 gap-4 min-h-0">
            <div className={`${buildType === 'web' ? 'lg:col-span-2' : 'lg:col-span-3'} min-h-0 flex flex-col gap-4`}>
                <FileExplorer files={filePaths} activeFile={activeFile} onSelectFile={setActiveFile} />
                <div className="bg-slate-800/50 p-3 rounded-lg border border-slate-700 flex-grow flex flex-col min-h-0">
                    <h3 className="text-sm font-bold text-slate-300 mb-2 px-1 flex-shrink-0">Build History</h3>
                    <div ref={historyContainerRef} className="overflow-y-auto space-y-2 text-sm">
                        {history.map((h, i) => (
                            <div key={i} className="text-slate-400 bg-slate-700/50 p-2 rounded-md">{h}</div>
                        ))}
                    </div>
                </div>
            </div>
            <div className={`${buildType === 'web' ? 'lg:col-span-5' : 'lg:col-span-9'} min-h-0 flex flex-col`}>
                <div className="bg-slate-800/50 p-1 rounded-lg border border-slate-700 h-full flex flex-col">
                    <div className="bg-slate-700/80 px-3 py-1.5 rounded-t-md">
                        <h3 className="text-sm font-bold text-slate-300">Editor: {activeFile}</h3>
                    </div>
                    <div className="flex-grow overflow-auto rounded-b-md">
                        <CodeBlock code={activeFileContent} />
                    </div>
                </div>
            </div>
            {buildType === 'web' && (
                <div className="lg:col-span-5 min-h-0">
                    <Preview files={files} />
                </div>
            )}
         </div>
      )}

      {!isLoading && files.length === 0 && !error && (
        <div className="flex-grow flex justify-center items-center">
            <div className="text-center text-slate-500">
                <svg xmlns="http://www.w3.org/2000/svg" className="mx-auto h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
                    <path d="M5 3a2 2 0 00-2 2v2a2 2 0 002 2h2a2 2 0 002-2V5a2 2 0 00-2-2H5zM5 11a2 2 0 00-2 2v2a2 2 0 002 2h2a2 2 0 002-2v-2a2 2 0 00-2-2H5zM11 5a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V5zM11 13a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
                </svg>
                <p className="mt-4">Describe the application you want to build and let the AI do the work.</p>
            </div>
        </div>
      )}
    </div>
  );
};

// =================================================================================
// --- ROOT APP COMPONENT ---
// =================================================================================

const App: React.FC = () => {
  const [activeTool, setActiveTool] = useState<AITool>(AITools[0]);

  const renderActiveTool = () => {
    switch (activeTool.id) {
      case 'chat':
        return <Chat />;
      case 'live-chat':
        return <LiveChat />;
      case 'image-generator':
        return <ImageGenerator />;
      case 'image-editor':
        return <ImageEditor />;
      case 'video-generator':
        return <VideoGenerator />;
      case 'web-search':
        return <WebSearch />;
      case 'json-generator':
        return <JsonGenerator />;
      case 'build':
        return <Build />;
      default:
        return <Chat />;
    }
  };

  return (
    <div className="min-h-screen text-white font-sans">
      <main className="container mx-auto px-4 py-8">
        <Header activeTool={activeTool} setActiveTool={setActiveTool} />
        <div className="mt-8 bg-slate-800 border border-slate-700 rounded-xl shadow-2xl overflow-hidden">
          {renderActiveTool()}
        </div>
      </main>
    </div>
  );
};


// =================================================================================
// --- RENDER APPLICATION ---
// =================================================================================
const rootElement = document.getElementById('root');
if (!rootElement) {
  throw new Error("Could not find root element to mount to");
}

const root = ReactDOM.createRoot(rootElement);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

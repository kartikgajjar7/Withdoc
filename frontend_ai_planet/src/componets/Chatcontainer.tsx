import Inputbox from "./Inputbox";
import { ChatMessage } from "./Chatmessage";
import React, { useState } from "react";

interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: string;
  isBotThinking?: boolean;
}

interface ChatContainerProps {
  fileid: string;
  messages: Message[];
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
  history: string[]; // Adjust the type based on what `history` actually contains
  sethistory: React.Dispatch<React.SetStateAction<string[]>>;
}

export default function ChatContainer({
  fileid,
  messages,
  setMessages,
  history,
  sethistory,
}: ChatContainerProps) {
  const [isLoading, setIsLoading] = useState(false);

  const handleSendMessage = async (text: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      text,
      isUser: true,
      timestamp: new Date().toLocaleTimeString(),
    };

    setMessages((prev) => [...prev, userMessage]);

    const thinkingMessage: Message = {
      id: `thinking-${Date.now()}`,
      text: "Thinking...",
      isUser: false,
      isBotThinking: true,
      timestamp: new Date().toLocaleTimeString(),
    };
    setMessages((prev) => [...prev, thinkingMessage]);

    setIsLoading(true);

    try {
      const response = await fetch(`http://127.0.0.1:8000/ask/${fileid}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: text, history: history }),
      });

      const data = await response.json();
      const responseMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: data.answer,
        isUser: false,
        timestamp: new Date().toLocaleTimeString(),
      };

      setMessages((prev) => [
        ...prev.filter((msg) => msg.id !== thinkingMessage.id),
        responseMessage,
      ]);

      sethistory((prev) => [...prev, data.history]);
    } catch (error) {
      console.error("Error sending message:", error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: "Sorry, there was an error processing your message.",
        isUser: false,
        timestamp: new Date().toLocaleTimeString(),
      };

      setMessages((prev) => [
        ...prev.filter((msg) => msg.id !== thinkingMessage.id),
        errorMessage,
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <main className="flex-1 overflow-y-auto p-4">
        <div className="">
          {messages.map((message) => (
            <ChatMessage
              key={message.id}
              message={message.text}
              isUser={message.isUser}
              timestamp={message.timestamp}
              isBotThinking={message.isBotThinking}
            />
          ))}
        </div>
      </main>

      <Inputbox onsendmessage={handleSendMessage} disabled={isLoading} />
    </div>
  );
}

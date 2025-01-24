import ReactMarkdown from "react-markdown";
interface ChatMessageProps {
  message: string;
  isUser: boolean;
  timestamp: string;
  isBotThinking?: boolean;
}
export function ChatMessage({
  message,
  isUser,
  timestamp,
  isBotThinking = false,
}: ChatMessageProps) {
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}>
      {!isUser && (
        <div className="w-8 h-8 rounded-full bg-gray-300 flex-shrink-0 mr-2">
          <img
            src="/bot.webp"
            alt="Bot Avatar"
            className="w-full h-full rounded-full"
          />
        </div>
      )}

      <div
        className={`max-w-[70%] rounded-lg p-4 ${
          isUser ? "bg-blue-500 text-white" : "bg-gray-100"
        }`}
      >
        {isBotThinking ? (
          <div className="flex items-center gap-2">
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-600"></div>
            <span>Thinking...</span>
          </div>
        ) : (
          <>
            <ReactMarkdown className="text-sm">{message}</ReactMarkdown>

            <p
              className={`text-xs mt-1 ${
                isUser ? "text-blue-100" : "text-gray-500"
              }`}
            >
              {timestamp}
            </p>
          </>
        )}
      </div>

      {isUser && (
        <div className="w-8 h-8 rounded-full bg-blue-300 flex-shrink-0 ml-2 overflow-hidden">
          <img
            src="/user.png"
            alt="User Avatar"
            className="w-full h-full object-cover"
          />
        </div>
      )}
    </div>
  );
}

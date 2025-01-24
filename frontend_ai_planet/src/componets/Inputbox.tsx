import React, { useState } from "react";
import { Send } from "lucide-react";

const Inputbox = ({ onsendmessage, disabled }) => {
  const [message, setMessage] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !disabled) {
      onsendmessage(message);
      setMessage("");
    }
  };

  return (
    <form onSubmit={handleSubmit} className="p-4">
      <div className="relative">
        <input
          type="text"
          value={message}
          disabled={disabled}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Send a message..."
          className="w-full px-4 py-3 pr-12 rounded-lg bg-gray-100 focus:outline-none focus:ring-2 focus:ring-emerald-500"
        />
        <button
          disabled={disabled || !message.trim()}
          type="submit"
          className="absolute right-2 top-1/2 -translate-y-1/2 p-2 text-gray-400 hover:text-emerald-500 transition-colors"
        >
          <Send size={20} />
        </button>
      </div>
    </form>
  );
};

export default Inputbox;

import Header from "./componets/Header";
import "./App.css";
import Chatcontainer from "./componets/Chatcontainer";
import { useState } from "react";

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [history, sethistory] = useState<Message[]>([]);
  const [file, setfile] = useState("");
  const [fileid, setfileid] = useState(null);
  return (
    <div className="flex flex-col h-screen bg-white">
      <Header sethistory={sethistory} setfile={setfile} setfileid={setfileid} />
      <div className="flex-1 min-h-0">
        <Chatcontainer
          messages={messages}
          setMessages={setMessages}
          sethistory={sethistory}
          history={history}
          fileid={fileid}
        />
      </div>
    </div>
  );
}

export default App;

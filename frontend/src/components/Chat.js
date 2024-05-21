import React, { useRef, useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import './Chat.css';

function Chat() {
  const {chat_id } = useParams();
  const [user_id] = useState(localStorage.getItem('username'));
  const [data, setData] = useState([]);
  const [chats, setChats] = useState([]);
  const [inputValue, setInputValue] = useState("");

  useEffect(() => {
    fetch(`http://localhost:8000/chat/${user_id}/${chat_id}`)
      .then(response => response.json())
      .then(data => setData(data[0].data));
    
    fetch(`http://localhost:8000/chats/${user_id}`)
      .then(response => response.json())
      .then(data => setChats(data.data));
  }, [user_id, chat_id]);

  const handleSend = () => {
    let new_question = { question: inputValue, type: 'sent' };
    setData([...data, new_question]);
    fetch('http://localhost:8000/query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text: inputValue }),
    })
    .then(response => response.json())
    .then(data => setData(data));
  };

  const handleKeyDown = (event) => {
    if(event.key === 'Enter'){
      handleSend();
    }
  }

  const messagesEndRef = useRef(null);


  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }

  useEffect(scrollToBottom, [data]);

  return (
  <div>
    <h1>Chat Page</h1>
    <div className="container">
      <div className="sidebar">
          <h2>Chats</h2>
          {chats.map((chat) => (
            <a href={`/chat/${chat.chat_id}`}>{chat.chat_id}</a>
          ))}
      </div>
      <div className='main-content'>
        <div className="chat-messages">
          {data.map((message, index) => (
            <p key={index} className={`message ${message.type}`}>
              {message.question}{message.answer}
              <br />
              {message.source && <span> Source: {message.source}</span>}
            </p>
          ))}
          <div ref={messagesEndRef} />
        </div>
        <div className='input-bar'>
          <input type="text"  className="input-bar" value={inputValue} onChange={e => setInputValue(e.target.value)} onKeyDown={handleKeyDown} />
          <button className="send-button" onClick={handleSend}>
              <span className="material-icons">send</span>
          </button>
        </div>
      </div>
    </div>
  </div>
  );
}

export default Chat;

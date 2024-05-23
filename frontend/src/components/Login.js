import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './Login.css';

function Login() {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const navigate = useNavigate();

    const handleSubmit = (event) => {
        if (username === '' || password === '') {
            alert('Please enter username and password');
            return;
        }
        event.preventDefault();
        fetch('http://localhost:8000/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ "id": username, "password": password }),
        })
            .then(console.log(body))
            .then(response => response.json())
            .then(data => {
                if (data) { 
                    localStorage.setItem('username', username);
                    navigate(data.redirect);
                }
                else{
                    alert("Invalid username or password");
                    setPassword('');
                    setUsername('');
                }
            });
    }

    const handleKeyDown = (event) => {
        if(event.key === 'Enter'){
          handleSubmit();
        }
      }

    return (
        <div className="login-container">
            <form onSubmit={handleSubmit}>
                <label>
                    Username: 
                    <input type="text" value={username} onChange={e => setUsername(e.target.value)} onKeyDown={handleKeyDown}/>
                </label>
                <label>
                    Password:
                    <input type="password" value={password} onChange={e => setPassword(e.target.value)} onKeyDown={handleKeyDown} />
                </label>
                <br/>
                <input type="submit" value="Submit" onClick={handleSubmit}/>
                <br/>
                <input type="submit" value="Register" onClick={() => navigate('/register')}/>
            </form>
        </div>
    );
}

export default Login;
import React, { useState } from 'react';
import './LoginPage.css';
import { useNavigate } from 'react-router-dom';

function Login() {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');

    const navigate = useNavigate();
    const handleLogin = async () => {
        console.log(username, password)
        fetch('http://localhost:8000/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ "id": username, "password": password }),
        })
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
    };

    const handleKeyDown = (event) => {
        if(event.key === 'Enter'){
            handleLogin();
        }
    }

    return (
        <div className="mainDiv">
            <button className="logo">PolitAI</button>
            <div className="wrapper">
                <input
                    type="text"
                    placeholder="Username"
                    className="username"
                    value={username}
                    onChange={(e) => {
                        setUsername(e.target.value)}}
                />
                <input
                    type="password"
                    placeholder="Password"
                    className="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                />
                <button className="login" onClick={handleLogin}>Login</button>
                <button className="register">Register</button>
            </div>
            <button className="guest">Continuar como convidado</button>
        </div>
    );
}

export default Login;

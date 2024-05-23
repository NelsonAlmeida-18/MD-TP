import React, { useState } from 'react';
import './LoginPage.css';

function Login() {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');

    const handleLogin = async () => {
        const response = await fetch('https://localhost:8000/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ username, password }),
        });

        const data = await response.json();
        console.log(data); // handle response
    };

    return (
        <div className="mainDiv">
            <button className="logo">PolitAI</button>
            <div className="wrapper">
                <input
                    type="text"
                    placeholder="Username"
                    className="username"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
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

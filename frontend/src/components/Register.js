import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './Login.css'; // Certifique-se de que o arquivo CSS está no mesmo diretório

function Register() {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const navigate = useNavigate();

    const handleSubmit = (event) => {
        event.preventDefault();
        if (email === '' || password === '') {
            alert('Please enter email and password');
            return;
        }
        fetch('http://localhost:8000/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({"id": email, "password": password }),
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    localStorage.setItem('username', email);
                    window.location.href = data.redirect;
                }
                else{
                    alert("Invalid email or password");
                    setPassword('');
                    setEmail('');
                }
            });
    };

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
                    <input type="text" value={email} onChange={e => setEmail(e.target.value)} onKeyDown={handleKeyDown} />
                </label>
                <br/>
                <label>
                    Password:
                    <input type="password" value={password} onChange={e => setPassword(e.target.value)} onKeyDown={handleKeyDown} />
                </label>
                <input type="submit" value="Register" onClick={handleSubmit}/>
                <br/>
                <input type="submit" value="Login" onClick={() => navigate('/login')}/>
            </form>
        </div>
    );
}

export default Register;
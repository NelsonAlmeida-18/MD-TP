import React from 'react';
import logo from "../imgs/politaiLogo.png"
import './LandingPage.css'

function LandingPage(){
    return(
        <div className="mainDiv">
            <div className="leftMember">
                <h1 className = "Logo">PolitAI</h1>
                <p className='paragraph'>Informar-se sobre os planos
                    eleitorais dos partidos 
                    portugueses nunca foi tão fácil...
                </p>
                <button className="launchapp" onClick={() => console.log("Clicked")}>Experimente Já</button>
            </div>

            <img src={logo} alt='PolitAI logo' className="rightMember" />

        </div>
    )
}

export default LandingPage;
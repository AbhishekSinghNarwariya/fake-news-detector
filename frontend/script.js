// ==========================================================
// AI Fake News Detector
// Premium Frontend - Part 1
// ==========================================================

// ---------------- DOM Elements ----------------

const newsInput = document.getElementById("newsInput");
const charCount = document.getElementById("charCount");

const predictBtn = document.getElementById("predictBtn");
const clearBtn = document.getElementById("clearBtn");
const sampleBtn = document.getElementById("sampleBtn");

const resultText = document.getElementById("resultText");
const resultCard = document.getElementById("resultCard");

const loading = document.getElementById("loading");

const copyBtn = document.getElementById("copyBtn");

const historyList = document.getElementById("historyList");

const themeToggle = document.getElementById("themeToggle");

const totalPredictions =
    document.getElementById("totalPredictions");

const fakeCount =
    document.getElementById("fakeCount");

const realCount =
    document.getElementById("realCount");

// ---------------- Character Counter ----------------

newsInput.addEventListener("input", () => {

    charCount.textContent =
        `Characters: ${newsInput.value.length}`;

});

// ---------------- Sample News ----------------

sampleBtn.addEventListener("click", () => {

    newsInput.value = `
Scientists discovered a new species of butterfly in the Amazon rainforest after years of research.
`;

    newsInput.dispatchEvent(new Event("input"));

});

// ---------------- Clear Button ----------------

clearBtn.addEventListener("click", () => {

    newsInput.value = "";

    charCount.textContent = "Characters: 0";

    resultCard.classList.remove(
        "result-success",
        "result-danger"
    );

    resultText.innerHTML =
        "Waiting for analysis...";

});

// ---------------- Dark Mode ----------------

// ---------------- Theme Toggle ----------------

loadTheme();

themeToggle.addEventListener("change", () => {

    if (themeToggle.checked) {

        document.body.classList.add("dark");

        localStorage.setItem("theme", "dark");

    } else {

        document.body.classList.remove("dark");

        localStorage.setItem("theme", "light");

    }

});

function loadTheme() {

    const savedTheme = localStorage.getItem("theme");

    if (savedTheme === "dark") {

        document.body.classList.add("dark");

        themeToggle.checked = true;

    } else {

        document.body.classList.remove("dark");

        themeToggle.checked = false;

    }

}

// ==========================================================
// AI Fake News Detector
// Premium Frontend - Part 2
// Prediction Logic
// ==========================================================

// ---------------- Predict Button ----------------

predictBtn.addEventListener("click", async () => {

    const news = newsInput.value.trim();

    if (!news) {

        resultCard.classList.remove(
            "result-success",
            "result-danger"
        );

        resultText.innerHTML =
            "⚠️ Please enter a news article.";

        return;

    }

    // Show Loading

    loading.classList.remove("hidden");

    resultCard.classList.add("hidden");

    try {

        const response = await fetch(
            "http://127.0.0.1:5000/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    news: news
                })
            }
        );

        const result = await response.json();

        loading.classList.add("hidden");

        resultCard.classList.remove("hidden");

        if (!response.ok) {

            resultText.innerHTML =
                `❌ ${result.message}`;

            return;

        }

        const prediction =
            result.data.prediction;

        const confidence =
            result.data.confidence;

        const processedText =
            result.data.processed_text;

        // Remove previous state

        resultCard.classList.remove(
            "result-success",
            "result-danger"
        );

        let barClass = "real";

        if (
            prediction
            .toLowerCase()
            .includes("fake")
        ) {

            resultCard.classList.add(
                "result-danger"
            );

            barClass = "fake";

        } else {

            resultCard.classList.add(
                "result-success"
            );

        }

        resultText.innerHTML = `

            <div class="result-label">

                ${prediction}

            </div>

            <div class="confidence">

                Confidence :
                ${confidence.toFixed(2)}%

            </div>

            <div class="progress">

                <div

                    class="progress-bar ${barClass}"

                    style="width:${confidence}%">

                </div>

            </div>

            <br>

            <strong>

                Processed Text

            </strong>

            <br><br>

            ${processedText}

        `;

        savePrediction(
            prediction,
            confidence
        );

    }

    catch(error){

        loading.classList.add("hidden");

        resultCard.classList.remove("hidden");

        resultText.innerHTML =

            "❌ Unable to connect to the backend server.";

        console.error(error);

    }

});

// ==========================================================
// AI Fake News Detector
// Premium Frontend - Part 3
// History, Dashboard & Copy Result
// ==========================================================

// ---------------- Save Prediction ----------------

function savePrediction(prediction, confidence) {

    const history =
        JSON.parse(
            localStorage.getItem("history")
        ) || [];

    history.unshift({

        prediction,

        confidence:
            confidence.toFixed(2),

        date:
            new Date().toLocaleString()

    });

    if(history.length > 5){

        history.pop();

    }

    localStorage.setItem(
        "history",
        JSON.stringify(history)
    );


    displayHistory();

    updateDashboard();

}

// ---------------- Display History ----------------

function displayHistory(){

    const history =
        JSON.parse(
            localStorage.getItem("history")
        ) || [];

    if(history.length === 0){

        historyList.innerHTML =
            "No predictions yet.";

        return;

    }

    historyList.innerHTML = "";

    history.forEach(item=>{

        const icon =
            item.prediction
            .toLowerCase()
            .includes("fake")
            ? "📰"
            : "✅";

        historyList.innerHTML += `

            <div class="history-item">

                <strong>

                    ${icon}
                    ${item.prediction}

                </strong>

                <br>

                Confidence :
                ${item.confidence}%

                <br>

                <small>

                    ${item.date}

                </small>

            </div>

        `;

    });

}

// ---------------- Dashboard ----------------

function updateDashboard(){

    const history =
        JSON.parse(
            localStorage.getItem("history")
        ) || [];

    totalPredictions.textContent =
        history.length;

    let fake = 0;
    let real = 0;

    history.forEach(item=>{

        if(
            item.prediction
            .toLowerCase()
            .includes("fake")
        ){

            fake++;

        }else{

            real++;

        }

    });

    fakeCount.textContent = fake;

    realCount.textContent = real;

}

// ---------------- Copy Result ----------------

copyBtn.addEventListener("click",()=>{

    const text =
        resultText.innerText;

    if(
        text ===
        "Waiting for analysis..."
    ){

        return;

    }

    navigator.clipboard
        .writeText(text);

    copyBtn.innerHTML =
        "✅ Copied!";

    setTimeout(()=>{

        copyBtn.innerHTML =
            "📋 Copy Result";

    },1500);

});

// ---------------- Initialize ----------------

displayHistory();

updateDashboard();

// ---------------- Theme Toggle Visibility ----------------

window.addEventListener("scroll", () => {

    const scrollTop = window.scrollY;

    const windowHeight = window.innerHeight;

    const documentHeight = document.documentElement.scrollHeight;

    const atTop = scrollTop < 30;

    const atBottom =
        scrollTop + windowHeight >= documentHeight - 30;

    if (atTop || atBottom) {

        document
            .querySelector(".theme-switch")
            .classList.remove("theme-hidden");

    } else {

        document
            .querySelector(".theme-switch")
            .classList.add("theme-hidden");

    }

});
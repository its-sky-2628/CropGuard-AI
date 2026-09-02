const input = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const uploadContent = document.getElementById("uploadContent");
const analyzeBtn = document.getElementById("analyzeBtn");
const result = document.getElementById("result");

input.addEventListener("change", () => {
    const file = input.files[0];

    if (!file) return;

    preview.src = URL.createObjectURL(file);
    preview.style.display = "block";
    uploadContent.style.display = "none";
    analyzeBtn.style.display = "inline-block";

    result.innerHTML = "";
});

analyzeBtn.addEventListener("click", async () => {
    const file = input.files[0];

    if (!file) {
        alert("Please select a plant image first.");
        return;
    }

    analyzeBtn.disabled = true;
    analyzeBtn.innerText = "🤖 CropGuard AI is analyzing...";

    result.innerHTML = `
        <div class="loading-result">
            <div class="loader"></div>
            <p>Analyzing plant image with AI...</p>
        </div>
    `;

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.message || "Prediction failed");
        }

        const confidence = Number(data.confidence).toFixed(2);

        let risk = "LOW";
        let riskClass = "risk-low";
        let riskText = "Crop appears stable. Continue regular monitoring.";

        const prediction = data.prediction.toLowerCase();

        if (
            prediction.includes("blight") ||
            prediction.includes("virus") ||
            prediction.includes("bacterial") ||
            prediction.includes("rust")
        ) {
            risk = "HIGH";
            riskClass = "risk-high";
            riskText = "Early inspection and appropriate management are recommended.";
        } else if (
            prediction.includes("spot") ||
            prediction.includes("mildew") ||
            prediction.includes("mite") ||
            prediction.includes("scorch") ||
            prediction.includes("mold")
        ) {
            risk = "MEDIUM";
            riskClass = "risk-medium";
            riskText = "Monitor the crop regularly and take preventive action.";
        }

        if (prediction.includes("healthy")) {
            risk = "LOW";
            riskClass = "risk-low";
            riskText = "No major disease symptoms detected by the AI model.";
        }

        result.innerHTML = `
            <div class="result-card">

                <div class="result-header">
                    <div>
                        <span class="result-label">CROPGUARD AI RESULT</span>
                        <h2>${data.prediction}</h2>
                    </div>

                    <div class="confidence-circle">
                        <strong>${confidence}%</strong>
                        <span>Confidence</span>
                    </div>
                </div>

                <div class="risk-box ${riskClass}">
                    <strong>⚠️ ${risk} RISK</strong>
                    <p>${riskText}</p>
                </div>

                <div class="result-details">
                    <div>
                        <span>🌿 Detected Condition</span>
                        <strong>${data.prediction}</strong>
                    </div>

                    <div>
                        <span>🎯 AI Confidence</span>
                        <strong>${confidence}%</strong>
                    </div>

                    <div>
                        <span>🚨 Risk Level</span>
                        <strong>${risk}</strong>
                    </div>
                </div>

                <div class="prototype-note">
                    ⚠️ AI prediction is based on the trained PlantVillage dataset.
                    For serious crop treatment decisions, consult an agricultural expert.
                </div>

            </div>
        `;

    } catch (error) {
        console.error(error);

        result.innerHTML = `
            <div class="result-card error-card">
                <h3>⚠️ Analysis Failed</h3>
                <p>${error.message}</p>
                <br>
                <small>
                    Make sure the CropGuard backend is running on port 8000.
                </small>
            </div>
        `;

    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.innerText = "Analyze With CropGuard AI";
    }
});

document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("predictForm");
    const submitBtn = document.getElementById("submitBtn");
    const btnText = submitBtn.querySelector(".btn-text");
    const btnLoader = submitBtn.querySelector(".btn-loader");
    const resultCard = document.getElementById("resultCard");
    const errorCard = document.getElementById("errorCard");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        hideResults();

        const formData = new FormData(form);
        const payload = Object.fromEntries(formData.entries());

        setLoading(true);

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            const data = await response.json();

            if (!response.ok) {
                showError(data.error || "Something went wrong.");
                return;
            }

            showResult(data);
        } catch {
            showError("Unable to connect to the server. Please try again.");
        } finally {
            setLoading(false);
        }
    });

    function setLoading(loading) {
        submitBtn.disabled = loading;
        btnText.classList.toggle("hidden", loading);
        btnLoader.classList.toggle("hidden", !loading);
    }

    function hideResults() {
        resultCard.classList.add("hidden");
        errorCard.classList.add("hidden");
    }

    function showResult(data) {
        document.getElementById("formattedPrice").textContent = data.formatted;
        document.getElementById("resultCurrency").textContent = data.currency;
        document.getElementById("resultInr").textContent = data.formatted_inr;

        const inrEl = document.getElementById("inrEquivalent");
        if (data.currency !== "INR") {
            inrEl.textContent = `≈ ${data.formatted_inr} (Indian Rupee)`;
            inrEl.classList.remove("hidden");
        } else {
            inrEl.textContent = "";
        }

        resultCard.classList.remove("hidden");
        resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    function showError(message) {
        document.getElementById("errorMessage").textContent = message;
        errorCard.classList.remove("hidden");
    }
});

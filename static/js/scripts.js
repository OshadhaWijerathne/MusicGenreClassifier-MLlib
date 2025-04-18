document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('classify-form');
    const lyricsInput = document.getElementById('lyrics-input');
    const resultsContainer = document.getElementById('results-container');
    const ensembleSpan = document.getElementById('ensemble-prediction');
    const gbtSpan = document.getElementById('gbt-prediction');
    const lrSpan = document.getElementById('lr-prediction');
    const nbSpan = document.getElementById('nb-prediction');
    const errorDiv = document.getElementById('error-message');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Reset UI
        resultsContainer.classList.add('hidden');
        errorDiv.classList.add('hidden');

        const lyrics = lyricsInput.value.trim();
        if (lyrics.length < 10) {
            errorDiv.textContent = "Please provide at least 10 characters of lyrics.";
            errorDiv.classList.remove('hidden');
            return;
        }

        try {
            const response = await fetch('/classify', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ lyrics })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Classification failed");
            }

            // Show predictions
            ensembleSpan.textContent = data.ensemble_prediction;
            gbtSpan.textContent = data.individual_predictions.gradient_boosted_trees;
            lrSpan.textContent = data.individual_predictions.logistic_regression;
            nbSpan.textContent = data.individual_predictions.naive_bayes;

            resultsContainer.classList.remove('hidden');

        } catch (err) {
            errorDiv.textContent = err.message;
            errorDiv.classList.remove('hidden');
        }
    });
});

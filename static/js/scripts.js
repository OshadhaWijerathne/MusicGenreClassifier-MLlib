document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('classify-form');
    const lyricsInput = document.getElementById('lyrics-input');
    const resultsContainer = document.getElementById('results-container');
    const ensembleSpan = document.getElementById('ensemble-prediction');
    const ensembleCard = document.getElementById('ensemble-card');
    const gbtSpan = document.getElementById('gbt-prediction');
    const lrSpan = document.getElementById('lr-prediction');
    const nbSpan = document.getElementById('nb-prediction');
    const errorDiv = document.getElementById('error-message');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = submitBtn.querySelector('.btn-text');

    const genreStyles = {
        'pop': { class: 'genre-pop', icon: '🎤' },
        'country': { class: 'genre-country', icon: '🤠' },
        'blues': { class: 'genre-blues', icon: '🎷' },
        'jazz': { class: 'genre-jazz', icon: '🎺' },
        'reggae': { class: 'genre-reggae', icon: '🌴' },
        'rock': { class: 'genre-rock', icon: '🎸' },
        'hip hop': { class: 'genre-hip-hop', icon: '🎵' }
    };

    function getGenreStyle(genre) {
        const lowerGenre = genre.toLowerCase();
        return genreStyles[lowerGenre] || { class: 'genre-pop', icon: '🎵' };
    }

    function showLoading() {
        submitBtn.disabled = true;
        btnText.innerHTML = '<div class="loading-spinner"></div>Analyzing...';
        resultsContainer.classList.add('hidden');
        resultsContainer.classList.remove('show');
        errorDiv.classList.add('hidden');
        errorDiv.classList.remove('show');
    }

    function hideLoading() {
        submitBtn.disabled = false;
        btnText.textContent = 'Classify Genre';
    }

    function showResults(data) {
        const ensembleStyle = getGenreStyle(data.ensemble_prediction);
        ensembleCard.className = `ensemble-result ${ensembleStyle.class}`;
        ensembleSpan.innerHTML = `${ensembleStyle.icon} ${data.ensemble_prediction}`;

        const gbtStyle = getGenreStyle(data.individual_predictions.gradient_boosted_trees);
        gbtSpan.innerHTML = `${gbtStyle.icon} ${data.individual_predictions.gradient_boosted_trees}`;

        const lrStyle = getGenreStyle(data.individual_predictions.logistic_regression);
        lrSpan.innerHTML = `${lrStyle.icon} ${data.individual_predictions.logistic_regression}`;

        const nbStyle = getGenreStyle(data.individual_predictions.naive_bayes);
        nbSpan.innerHTML = `${nbStyle.icon} ${data.individual_predictions.naive_bayes}`;

        resultsContainer.classList.remove('hidden');
        setTimeout(() => resultsContainer.classList.add('show'), 100);
    }

    function showError(message) {
        errorDiv.textContent = message;
        errorDiv.classList.remove('hidden');
        setTimeout(() => errorDiv.classList.add('show'), 100);
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const lyrics = lyricsInput.value.trim();
        if (lyrics.length < 10) {
            showError("Please provide at least 10 characters of lyrics.");
            return;
        }
        showLoading();

        try {
            const response = await fetch('/classify', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ lyrics })
            });

            const data = await response.json();

            if (!response.ok) throw new Error(data.error || "Classification failed");

            setTimeout(() => {
                hideLoading();
                showResults(data);
            }, 800);

        } catch (err) {
            hideLoading();
            showError(err.message);
        }
    });

    // Demo mode for testing (remove in production)
    if (!window.location.href.includes('127.0.0.1:5000')) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const lyrics = lyricsInput.value.trim();
            if (lyrics.length < 10) {
                showError("Please provide at least 10 characters of lyrics.");
                return;
            }
            showLoading();

            const genres = ['pop', 'country', 'blues', 'jazz', 'reggae', 'rock', 'hip hop'];
            const randomGenre = genres[Math.floor(Math.random() * genres.length)];
            const randomGenre2 = genres[Math.floor(Math.random() * genres.length)];
            const randomGenre3 = genres[Math.floor(Math.random() * genres.length)];

            setTimeout(() => {
                hideLoading();
                const demoData = {
                    ensemble_prediction: randomGenre,
                    individual_predictions: {
                        gradient_boosted_trees: randomGenre,
                        logistic_regression: randomGenre2,
                        naive_bayes: randomGenre3
                    }
                };
                showResults(demoData);
            }, 1500);
        });
    }
});

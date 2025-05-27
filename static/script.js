document.getElementById('predict-form').addEventListener('submit', async function (e) {
  e.preventDefault();

  const text = document.getElementById('text-input').value.trim();
  const resultDiv = document.getElementById('result');
  const errorMessage = document.getElementById('error-message');
  const assistant = document.getElementById('assistant');

  resultDiv.classList.add('hidden');
  errorMessage.classList.add('hidden');
  errorMessage.classList.remove('animate__shakeX');

  // Show assistant animation
  assistant.style.display = 'flex';

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || 'Unexpected error occurred');
    }

    document.getElementById('prediction').textContent = data.prediction;
    document.getElementById('confidence').textContent = data.confidence;
    resultDiv.classList.remove('hidden');

  } catch (error) {
    errorMessage.textContent = error.message;
    errorMessage.classList.remove('hidden');

    void errorMessage.offsetWidth; // restart animation
    errorMessage.classList.add('animate__shakeX');
  } finally {
    // Hide assistant after analysis
    setTimeout(() => {
      assistant.style.display = 'none';
    }, 1500);
  }
});

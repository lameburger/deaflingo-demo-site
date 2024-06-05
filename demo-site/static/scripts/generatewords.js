// List of random words
const words = ['hello', 'thanks', 'iloveyou'];

// Function to generate a random word
function generateRandomWord() {
    const randomIndex = Math.floor(Math.random() * words.length);
    return words[randomIndex];
}

// Populate the word container with random words
const wordContainer = document.querySelector('.wordcontainer .word');
for (let i = 0; i < 5; i++) { // Populate 3 words
    const word = generateRandomWord();
    const spans = Array.from(word).map(letter => `<span>${letter}</span>`).join('');
    wordContainer.innerHTML += `<div>${spans}</div>`;
}

const wordSpans = document.querySelectorAll('.word span');

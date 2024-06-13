// List of random words
let words = ['burrito', 'hello', 'i love you', 'meet you', 'my', 'name', 'nice', 'thanks', 'what', 'where'];

// Function to generate a random word without repetition
function generateRandomWord() {
    const randomIndex = Math.floor(Math.random() * words.length);
    const word = words[randomIndex];
    words.splice(randomIndex, 1); // Remove the selected word from the array
    return word;
}

// Populate the word container with random words
const wordContainer = document.querySelector('.wordcontainer .word');
for (let i = 0; i < 5; i++) { // Populate 5 words
    if (words.length === 0) break; // Break if no more words are available
    const word = generateRandomWord();
    const spans = Array.from(word).map(letter => `<span>${letter}</span>`).join('');
    wordContainer.innerHTML += `<div>${spans}</div>`;
}

const wordSpans = document.querySelectorAll('.word span');

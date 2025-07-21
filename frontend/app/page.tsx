'use client';

import { useState } from 'react';

export default function HomePage() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);

  const askQuestion = async () => {
    if (!question.trim()) return;

    setLoading(true);
    setAnswer('');

    try {
      const res = await fetch('http://localhost:8000/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      });

      const data = await res.json();
      setAnswer(data.answer || 'Keine Antwort erhalten.');
    } catch (err) {
      console.error(err);
      setAnswer('Fehler beim Senden der Anfrage.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="flex flex-col items-center justify-center min-h-screen p-4 bg-gray-50">
      <div className="w-full max-w-2xl space-y-4">
        <h1 className="text-3xl font-bold text-center text-black">🎓 MMI RAG Chatbot</h1>
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Stelle eine Frage zum Modulhandbuch..."
          rows={4}
          className="w-full p-3 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 text-black"
        />

        <button
          onClick={askQuestion}
          disabled={loading}
          className="w-full py-2 text-white bg-blue-600 rounded hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? 'Antwort wird generiert...' : 'Frage stellen'}
        </button>

        {answer && (
          <div className="p-4 mt-4 bg-white border rounded shadow">
            <h2 className="mb-2 text-lg font-semibold">Antwort:</h2>
            <p className="whitespace-pre-wrap text-black">{answer}</p>
          </div>
        )}
      </div>
    </main>
  );
}

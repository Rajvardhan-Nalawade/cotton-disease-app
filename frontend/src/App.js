import React, { useState } from "react";

function App() {
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);
  const [dark, setDark] = useState(false);

  const toggleDark = () => {
    setDark(!dark);
    document.documentElement.classList.toggle("dark");
  };

  const processFile = async (file) => {
  setPreview(URL.createObjectURL(file));
  setLoading(true);
  setResult("");

  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch("http://127.0.0.1:8000/predict/", {
    method: "POST",
    body: formData,
  });

  const data = await res.json();
  setResult(`${data.class} (${(data.confidence * 100).toFixed(2)}%)`);
  setLoading(false);
};

const handleUpload = (e) => {
  const file = e.target.files[0];
  if (!file) return;
  processFile(file);
};
const handleDrop = (e) => {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  if (!file) return;
  processFile(file);
};

const handleDragOver = (e) => {
  e.preventDefault();
};




  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-6 bg-gradient-to-br from-green-100 to-green-50 dark:from-gray-900 dark:to-gray-800 transition-colors">

      {/* Dark toggle */}
      <button
        onClick={toggleDark}
        className="absolute top-4 right-4 px-4 py-2 rounded-xl bg-white/80 dark:bg-gray-700 text-sm shadow"
      >
        {dark ? "☀ Light" : "🌙 Dark"}
      </button>

      {/* Main card */}
      <div className="bg-white dark:bg-gray-900 rounded-3xl shadow-2xl p-8 max-w-lg w-full text-center transition-all">

        <h1 className="text-3xl font-extrabold text-green-700 dark:text-green-400 mb-2">
          Cotton Disease Detector
        </h1>

        <p className="text-gray-500 dark:text-gray-400 mb-6">
          Upload a Cotton leaf image to detect disease instantly
        </p>

        {/* Upload zone */}
        {/* Upload zone */}
<div
  onDrop={handleDrop}
  onDragOver={handleDragOver}
  className="cursor-pointer block border-2 border-dashed border-green-300 dark:border-gray-600 rounded-2xl p-6 hover:bg-green-50 dark:hover:bg-gray-800 transition"
>

  <input type="file" hidden onChange={handleUpload} />

  {!preview && (
    <p className="text-gray-500 dark:text-gray-400">
      Drag & drop an image here or click to upload
    </p>
  )}

  {preview && (
    <img
      src={preview}
      alt="preview"
      className="mx-auto h-56 object-cover rounded-xl"
    />
  )}

</div>


        {/* Loading */}
        {loading && (
          <div className="mt-6 animate-pulse text-blue-500 font-medium">
            Analyzing image...
          </div>
        )}

        {/* Result */}
        {result && (
          <div className="mt-6 text-xl font-bold text-green-700 dark:text-green-300">
            {result}
          </div>
        )}
      </div>

      {/* Footer */}
      <p className="mt-8 text-sm text-gray-400">
        AI-powered Cotton disease recognition
      </p>
    </div>
  );
}

export default App;

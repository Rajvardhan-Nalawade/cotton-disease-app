import React, { useState, useRef } from "react";

function App() {
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);
  const [dark, setDark] = useState(false);

  const toggleDark = () => {
    setDark(!dark);
    document.documentElement.classList.toggle("dark");
  };
  const fileInputRef = useRef(null);


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
  fileInputRef.current.value = null;
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

      <button
  onClick={toggleDark}
  className="absolute top-4 right-4 w-16 h-8 flex items-center bg-gray-300 dark:bg-gray-600 rounded-full p-1 transition"
>
  <div
    className={`w-6 h-6 bg-white rounded-full shadow-md transform transition ${
      dark ? "translate-x-8" : "translate-x-0"
    }`}
  />
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
<div
  onDrop={handleDrop}
  onDragOver={handleDragOver}
  className="border-2 border-dashed border-green-300 dark:border-gray-600 rounded-2xl p-6 mb-4 hover:bg-green-50 dark:hover:bg-gray-800 transition"
>


  <input
  type="file"
  hidden
  ref={fileInputRef}
  onChange={handleUpload}
/>


  {!preview && (
    <p className="text-gray-500 dark:text-gray-400">
      Drag & drop an image here
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
<button
  onClick={() => fileInputRef.current.click()}
  className="bg-green-600 hover:bg-green-700 text-white px-6 py-2 rounded-xl font-medium transition"
>
  Choose Image
</button>



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

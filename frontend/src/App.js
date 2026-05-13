  import React, { useState, useRef, useEffect } from "react";
  import diseaseInfo from "./data/diseaseinfo";

  function App() {
    const API_URL = process.env.REACT_APP_API_URL;

    const [preview, setPreview] = useState(null);
    const [loading, setLoading] = useState(false);
    const [dark, setDark] = useState(
      window.matchMedia("(prefers-color-scheme: dark)").matches
    );
    const [predictionData, setPredictionData] = useState(null);

    useEffect(() => {
      if (dark) {
        document.documentElement.classList.add("dark");
      } else {
        document.documentElement.classList.remove("dark");
      }
    }, [dark]);

    const toggleDark = () => {
      setDark(!dark);
    };
    const fileInputRef = useRef(null);


    const processFile = async (file) => {
    setPreview(URL.createObjectURL(file));
    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(`${API_URL}/predict/`, {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    setPredictionData({
      disease: data.class,
      confidence: (data.confidence * 100).toFixed(2)
    });
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
      <div
        onDrop={handleDrop} 
      onDragOver={handleDragOver} 
      className="w-full min-h-[100dvh] flex flex-col items-center px-6 py-10 bg-gradient-to-br from-green-100 to-green-50 dark:from-gray-900 dark:to-gray-800 transition-colors">

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
    Upload Image
  </button>



          {/* Loading */}
          {loading && (
            <div className="mt-6 animate-pulse text-blue-500 font-medium">
              Analyzing image...
            </div>
          )}

          
        </div>

        {predictionData && (
          <div className="mt-8 w-full max-w-7xl grid grid-cols-1 lg:grid-cols-3 gap-6">

            {/* Main Prediction Card */}
            <div className="lg:col-span-2 bg-white dark:bg-gray-900 rounded-3xl shadow-xl p-6">

              <h2 className="text-2xl font-bold text-green-700 dark:text-green-400 mb-2">
                {predictionData.disease}
              </h2>

              <p className="text-lg font-semibold mb-4 text-gray-700 dark:text-gray-300">
                Confidence: {predictionData.confidence}%
              </p>

              {/* Confidence Bar */}
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-4 overflow-hidden mb-6">
                <div
                  className="bg-green-600 h-4 rounded-full"
                  style={{ width: `${predictionData.confidence}%` }}
                />
              </div>

              {/* Top Sections */}
              <div className="space-y-6">

                {/* Description */}
                <div className="bg-green-50 dark:bg-gray-800 rounded-2xl p-5">
                  <h3 className="font-bold text-lg mb-2 text-green-700 dark:text-green-400">
                    About This Disease
                  </h3>

                  <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
                    {diseaseInfo[predictionData.disease].description}
                  </p>
                </div>

                {/* Symptoms */}
                <div className="bg-white dark:bg-gray-800 border dark:border-gray-700 rounded-2xl p-5">
                  <h3 className="font-bold text-lg mb-3 text-yellow-600">
                    Symptoms
                  </h3>

                  <ul className="list-disc ml-6 space-y-2 text-gray-700 dark:text-gray-300">
                    {diseaseInfo[predictionData.disease].symptoms.map((item, idx) => (
                      <li key={idx}>{item}</li>
                    ))}
                  </ul>
                </div>

                {/* Causes */}
                <div className="bg-white dark:bg-gray-800 border dark:border-gray-700 rounded-2xl p-5">
                  <h3 className="font-bold text-lg mb-3 text-red-600">
                    Causes
                  </h3>

                  <ul className="list-disc ml-6 space-y-2 text-gray-700 dark:text-gray-300">
                    {diseaseInfo[predictionData.disease].causes.map((item, idx) => (
                      <li key={idx}>{item}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>

            {/* Side Information */}
            <div className="space-y-6">

              {/* At A Glance */}
              <div className="bg-green-50 dark:bg-gray-900 rounded-3xl shadow-xl p-6">
                <h3 className="text-xl font-bold mb-4 text-green-700 dark:text-green-400">
                  At a Glance
                </h3>

                <div className="space-y-4 text-gray-700 dark:text-gray-300">

                  <div className="flex justify-between">
                    <span>Disease</span>
                    <span className="font-semibold">{predictionData.disease}</span>
                  </div>

                  <div className="flex justify-between">
                    <span>Severity</span>

                    <span className={`px-3 py-1 rounded-full text-white text-sm font-semibold ${
                      diseaseInfo[predictionData.disease].severity === "High"
                        ? "bg-red-500"
                        : diseaseInfo[predictionData.disease].severity === "Medium"
                        ? "bg-yellow-500"
                        : "bg-green-500"
                    }`}>
                      {diseaseInfo[predictionData.disease].severity}
                    </span>
                  </div>

                  <div className="flex justify-between">
                    <span>Confidence</span>
                    <span>{predictionData.confidence}%</span>
                  </div>
                </div>
              </div>

              {/* Remedy */}
              <div className="bg-white dark:bg-gray-900 rounded-3xl shadow-xl p-6">
                <h3 className="text-xl font-bold mb-4 text-green-700 dark:text-green-400">
                  Recommended Remedy
                </h3>

                <ul className="list-disc ml-6 space-y-2 text-gray-700 dark:text-gray-300">
                  {diseaseInfo[predictionData.disease].remedy.map((item, idx) => (
                    <li key={idx}>{item}</li>
                  ))}
                </ul>
              </div>

              {/* Prevention */}
              <div className="bg-white dark:bg-gray-900 rounded-3xl shadow-xl p-6">
                <h3 className="text-xl font-bold mb-4 text-green-700 dark:text-green-400">
                  Prevention Tips
                </h3>

                <ul className="list-disc ml-6 space-y-2 text-gray-700 dark:text-gray-300">
                  {diseaseInfo[predictionData.disease].prevention.map((item, idx) => (
                    <li key={idx}>{item}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}


        {/* Footer */}
        <p className="mt-8 text-sm text-gray-400">
          AI-powered Cotton disease recognition
        </p>
      </div>
    );
  }

  export default App;

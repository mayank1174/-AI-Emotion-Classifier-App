import { useState, useEffect, useRef } from "react";
import { Mic, MicOff, Type, Sparkles, Combine, Loader2, Zap, Activity, Camera, Video, VideoOff } from "lucide-react";
import { Button } from "./ui/button";
import { Textarea } from "./ui/textarea";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { EmotionResults } from "./EmotionResults";
import { Badge } from "./ui/badge";
import { motion, AnimatePresence } from "framer-motion";

export interface EmotionScore {
  emotion: string;
  score: number;
  color: string;
}

async function fetchClassify(text: string): Promise<EmotionScore[]> {
  const res = await fetch('/api/classify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error('Classification failed');
  return res.json();
}

async function fetchClassifyImage(imageBlob: Blob): Promise<EmotionScore[]> {
  const formData = new FormData();
  formData.append('image', imageBlob, 'capture.jpg');
  const res = await fetch('/api/classify_image', {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error('Image Classification failed');
  return res.json();
}

async function fetchCombine(...sets: (EmotionScore[] | null)[]): Promise<EmotionScore[]> {
  const validSets = sets.filter(Boolean) as EmotionScore[][];
  if (validSets.length < 1) return [];
  
  const res = await fetch('/api/combine', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ emotions1: validSets[0], emotions2: validSets[1] || [], emotions3: validSets[2] || [] }),
  });
  if (!res.ok) throw new Error('Combine failed');
  return res.json();
}

export function EmotionClassifier() {
  const [textInput, setTextInput] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  
  const [isLoadingText, setIsLoadingText] = useState(false);
  const [isLoadingSpeech, setIsLoadingSpeech] = useState(false);
  const [isLoadingFace, setIsLoadingFace] = useState(false);
  
  const [transcript, setTranscript] = useState("");
  const [textEmotions, setTextEmotions] = useState<EmotionScore[] | null>(null);
  const [speechEmotions, setSpeechEmotions] = useState<EmotionScore[] | null>(null);
  const [faceEmotions, setFaceEmotions] = useState<EmotionScore[] | null>(null);
  const [combinedEmotions, setCombinedEmotions] = useState<EmotionScore[] | null>(null);
  
  const [recognition, setRecognition] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  // Camera State
  const [isCameraActive, setIsCameraActive] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Initialize Speech Recognition
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      if (SpeechRecognition && !recognition) {
        const recognitionInstance = new SpeechRecognition();
        recognitionInstance.continuous = true;
        recognitionInstance.interimResults = true;
        recognitionInstance.lang = 'en-US';

        recognitionInstance.onstart = () => {
          setIsRecording(true);
          setTranscript("");
          setSpeechEmotions(null);
        };

        recognitionInstance.onresult = (event: any) => {
          let interimTranscript = '';
          let finalTranscript = '';

          for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcriptPiece = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
              finalTranscript += transcriptPiece + ' ';
            } else {
              interimTranscript += transcriptPiece;
            }
          }
          setTranscript((prev) => prev + finalTranscript || interimTranscript);
        };

        recognitionInstance.onerror = (event: any) => {
          console.error('Speech recognition error:', event.error);
          setIsRecording(false);
        };

        recognitionInstance.onend = () => {
          setIsRecording(false);
        };

        setRecognition(recognitionInstance);
      }
    }
  }, []);

  // Cleanup Camera on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  const updateCombinations = async (t: EmotionScore[] | null, s: EmotionScore[] | null, f: EmotionScore[] | null) => {
    try {
      const validEmotions = [t, s, f].filter(Boolean);
      if (validEmotions.length > 0) {
        const combined = await fetchCombine(t, s, f);
        setCombinedEmotions(combined);
      } else {
        setCombinedEmotions(null);
      }
    } catch {
      console.error("Failed to combine emotions");
    }
  };

  const analyzeText = async () => {
    if (!textInput.trim()) return;
    setIsLoadingText(true);
    setError(null);
    try {
      const results = await fetchClassify(textInput);
      setTextEmotions(results);
      await updateCombinations(results, speechEmotions, faceEmotions);
    } catch {
      setError('Could not reach the backend. Make sure app.py is running (python app.py).');
    } finally {
      setIsLoadingText(false);
    }
  };

  const startRecording = () => {
    if (!recognition) {
      alert('Speech recognition is not supported in your browser. Please use Chrome or Edge.');
      return;
    }
    recognition.start();
  };

  const stopRecording = async () => {
    if (recognition) {
      recognition.stop();
      setIsRecording(false);

      if (transcript.trim()) {
        setIsLoadingSpeech(true);
        setError(null);
        try {
          const results = await fetchClassify(transcript);
          setSpeechEmotions(results);
          await updateCombinations(textEmotions, results, faceEmotions);
        } catch {
          setError('Could not reach the backend.');
        } finally {
          setIsLoadingSpeech(false);
        }
      }
    }
  };

  const startCamera = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsCameraActive(true);
      }
    } catch (err) {
      setError("Camera access denied or unavailable.");
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setIsCameraActive(false);
    }
  };

  const captureAndAnalyzeFace = async () => {
    if (videoRef.current && canvasRef.current && isCameraActive) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(async (blob) => {
          if (blob) {
             setIsLoadingFace(true);
             setError(null);
             try {
                const results = await fetchClassifyImage(blob);
                setFaceEmotions(results);
                await updateCombinations(textEmotions, speechEmotions, results);
             } catch (err) {
                setError('Could not reach backend for face analysis.');
             } finally {
                setIsLoadingFace(false);
             }
          }
        }, 'image/jpeg');
      }
    }
  };

  const clearAll = () => {
    setTextInput("");
    setTranscript("");
    setTextEmotions(null);
    setSpeechEmotions(null);
    setFaceEmotions(null);
    setCombinedEmotions(null);
    setError(null);
    if (recognition && isRecording) {
      recognition.stop();
    }
    stopCamera();
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.3 }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 15 },
    show: { opacity: 1, y: 0, transition: { duration: 0.8, ease: "easeOut" as const } }
  };

  return (
    <div className="w-full relative py-12">
      {/* ── Hero / Title Section ── */}
      <motion.div
        className="flex flex-col items-center justify-center min-h-[30vh] text-center px-6 relative"
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1, ease: "easeOut" }}
      >
        <div className="flex items-center gap-3 mb-4 relative z-10 cursor-default">
          <Sparkles className="w-10 h-10 text-primary drop-shadow-sm opacity-80" />
          <motion.h1
            className="text-6xl md:text-7xl font-black tracking-tight leading-tight bg-clip-text text-transparent bg-gradient-to-r from-primary via-purple-500 to-secondary animate-gradient-text pb-2 px-4"
            whileHover={{ opacity: 0.8, scale: 1.001 }}
            transition={{ duration: 0.6, ease: "easeInOut" }}
          >
            Multimodal Emotion AI
          </motion.h1>
          <Sparkles className="w-10 h-10 text-secondary drop-shadow-sm opacity-80" />
        </div>

        <motion.p
          className="text-muted-foreground text-lg md:text-xl font-medium max-w-2xl px-4 relative z-10"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1.2, delay: 0.4 }}
        >
          Analyze emotions from text, speech, and facial expressions simultaneously for deeper, comprehensive insights.
        </motion.p>
      </motion.div>

      {/* ── Main Content ── */}
      <motion.div
        className="max-w-6xl mx-auto px-6 pb-8 mt-8"
        variants={containerVariants}
        initial="hidden"
        animate="show"
      >
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mb-6 p-4 bg-destructive/10 border border-destructive/20 rounded-xl text-destructive text-sm font-medium text-center"
            >
              ⚠️ {error}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Input Section — 3 cards side by side */}
        <div className="grid lg:grid-cols-3 md:grid-cols-2 gap-6 mb-8">

          {/* 1. Text Input Card */}
          <motion.div variants={itemVariants} whileHover={{ y: -4 }} className="flex">
            <Card className="flex flex-col w-full h-full border-border/50 shadow-xl shadow-primary/5 dark:shadow-primary/10 rounded-2xl overflow-hidden backdrop-blur-xl bg-card transition-all">
              <CardHeader className="pb-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="p-2.5 rounded-xl bg-primary/10 text-primary">
                      <Type className="w-5 h-5" />
                    </div>
                    <CardTitle className="text-lg">Text Analysis</CardTitle>
                  </div>
                  {textEmotions && (
                    <Badge variant="secondary" className="bg-green-100 text-green-700 dark:bg-green-500/10 dark:text-green-400">
                      Analyzed
                    </Badge>
                  )}
                </div>
                <CardDescription className="text-sm pt-2">
                  Type or paste text to detect semantic emotions.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4 flex-1 flex flex-col justify-end">
                <Textarea
                  placeholder="E.g., I'm feeling incredibly inspired today!"
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  className="h-36 mb-4 resize-none rounded-xl border-input focus:border-primary focus:ring-1 focus:ring-primary/50 text-base"
                />
                <Button
                  onClick={analyzeText}
                  disabled={!textInput.trim() || isLoadingText}
                  className="w-full bg-gradient-to-r from-primary to-purple-500 hover:from-primary/90 hover:to-purple-500/90 text-white rounded-xl shadow-lg shadow-primary/25 h-12 text-base font-semibold"
                >
                  {isLoadingText ? (
                    <><Loader2 className="w-5 h-5 mr-2 animate-spin" /> Analyzing...</>
                  ) : (
                    <><Zap className="w-5 h-5 mr-2" /> Analyze Text</>
                  )}
                </Button>
              </CardContent>
            </Card>
          </motion.div>

          {/* 2. Speech Input Card */}
          <motion.div variants={itemVariants} whileHover={{ y: -4 }} className="flex">
            <Card className="flex flex-col w-full h-full border-border/50 shadow-xl shadow-secondary/5 dark:shadow-secondary/10 rounded-2xl overflow-hidden backdrop-blur-xl bg-card transition-all">
              <CardHeader className="pb-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="p-2.5 rounded-xl bg-secondary/10 text-secondary">
                      <Mic className="w-5 h-5" />
                    </div>
                    <CardTitle className="text-lg">Speech Analysis</CardTitle>
                  </div>
                  {speechEmotions && (
                    <Badge variant="secondary" className="bg-green-100 text-green-700 dark:bg-green-500/10 dark:text-green-400">
                      Analyzed
                    </Badge>
                  )}
                </div>
                <CardDescription className="text-sm pt-2">
                  Record your voice to analyze vocal sentiment.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4 flex-1 flex flex-col justify-end">
                <div className="h-36 mb-4 relative flex flex-col items-center justify-center gap-3 rounded-xl border border-input bg-muted/30 overflow-hidden">
                  {isRecording ? (
                    <div className="flex flex-col items-center w-full px-6 z-10">
                      <div className="flex items-center gap-2 text-secondary font-semibold mb-3">
                        <Activity className="w-5 h-5 animate-pulse" /> Listening...
                      </div>
                      <p className="text-xs text-foreground/80 font-medium text-center line-clamp-2 w-full">
                        {transcript || 'Speak now...'}
                      </p>
                    </div>
                  ) : transcript ? (
                    <div className="p-4 w-full h-full text-sm text-foreground/80 text-center overflow-y-auto">
                      " {transcript} "
                    </div>
                  ) : (
                    <div className="flex flex-col items-center opacity-40">
                      <Mic className="w-8 h-8 mb-2" />
                      <p className="text-xs font-medium">Ready to record</p>
                    </div>
                  )}
                </div>

                {!isRecording ? (
                  <Button
                    onClick={startRecording}
                    className="w-full bg-gradient-to-r from-secondary to-pink-500 hover:from-secondary/90 hover:to-pink-500/90 text-white rounded-xl shadow-lg shadow-secondary/25 h-12 text-base font-semibold"
                  >
                    <Mic className="w-5 h-5 mr-2" /> Start Recording
                  </Button>
                ) : (
                  <Button
                    onClick={stopRecording}
                    variant="outline"
                    className="w-full border-secondary/50 text-secondary hover:bg-secondary/10 hover:text-secondary rounded-xl h-12 text-base font-semibold"
                    disabled={isLoadingSpeech}
                  >
                    {isLoadingSpeech ? (
                      <><Loader2 className="w-5 h-5 mr-2 animate-spin" /> Analyzing...</>
                    ) : (
                      <><MicOff className="w-5 h-5 mr-2" /> Stop & Analyze</>
                    )}
                  </Button>
                )}
              </CardContent>
            </Card>
          </motion.div>

          {/* 3. Facial Expression Card */}
          <motion.div variants={itemVariants} whileHover={{ y: -4 }} className="flex">
            <Card className="flex flex-col w-full h-full border-border/50 shadow-xl shadow-blue-500/5 dark:shadow-blue-500/10 rounded-2xl overflow-hidden backdrop-blur-xl bg-card transition-all">
              <CardHeader className="pb-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="p-2.5 rounded-xl bg-blue-500/10 text-blue-500">
                      <Camera className="w-5 h-5" />
                    </div>
                    <CardTitle className="text-lg">Facial Expression</CardTitle>
                  </div>
                  {faceEmotions && (
                    <Badge variant="secondary" className="bg-green-100 text-green-700 dark:bg-green-500/10 dark:text-green-400">
                      Analyzed
                    </Badge>
                  )}
                </div>
                <CardDescription className="text-sm pt-2">
                  Use your webcam to analyze facial expressions.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4 flex-1 flex flex-col justify-end">
                
                <div className="h-36 mb-4 relative flex flex-col items-center justify-center gap-3 rounded-xl border border-input bg-card bg-black/5 overflow-hidden group">
                  <div className={`flex flex-col items-center opacity-40 ${isCameraActive ? 'hidden' : ''}`}>
                     <VideoOff className="w-8 h-8 mb-2" />
                     <p className="text-xs font-medium">Camera off</p>
                  </div>
                  
                  <video 
                    ref={videoRef} 
                    autoPlay 
                    playsInline 
                    muted 
                    className={`w-full h-full object-cover rounded-xl ${!isCameraActive ? 'hidden' : ''}`} 
                  />
                  <canvas ref={canvasRef} className="hidden" />
                  
                  {isCameraActive && isLoadingFace && (
                    <div className="absolute top-0 w-full h-1 bg-blue-500/50 blur-sm animate-pulse shadow-[0_0_15px_3px_rgba(59,130,246,0.5)]" style={{ animation: 'scan 2s linear infinite' }} />
                  )}
                </div>

                {!isCameraActive ? (
                  <Button
                    onClick={startCamera}
                    className="w-full bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 text-white rounded-xl shadow-lg shadow-blue-500/25 h-12 text-base font-semibold"
                  >
                    <Video className="w-5 h-5 mr-2" /> Turn On Camera
                  </Button>
                ) : (
                  <div className="flex gap-2 w-full">
                    <Button
                      onClick={captureAndAnalyzeFace}
                      className="flex-1 bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 text-white rounded-xl h-12 text-base font-semibold"
                      disabled={isLoadingFace}
                    >
                      {isLoadingFace ? (
                        <Loader2 className="w-5 h-5 animate-spin" />
                      ) : (
                        <><Camera className="w-5 h-5 mr-1" /> Capture / Analyze</>
                      )}
                    </Button>
                    <Button onClick={stopCamera} variant="outline" className="w-12 h-12 p-0 rounded-xl border-destructive/20 text-destructive hover:bg-destructive/10">
                       <VideoOff className="w-5 h-5" />
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>

        </div>

        {/* Clear Button */}
        <AnimatePresence>
          {(textEmotions || speechEmotions || faceEmotions) && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="flex justify-center mb-10"
            >
              <Button onClick={clearAll} variant="outline" className="rounded-full px-8 shadow-sm">
                Clear Results
              </Button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Results Section */}
        <motion.div variants={containerVariants} className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-2">
          {textEmotions && (
            <motion.div variants={itemVariants}>
              <EmotionResults emotions={textEmotions} inputText={textInput} title="Text Analysis" icon={Type} />
            </motion.div>
          )}

          {speechEmotions && (
            <motion.div variants={itemVariants}>
              <EmotionResults emotions={speechEmotions} inputText={transcript} title="Speech Analysis" icon={Mic} />
            </motion.div>
          )}
          
          {faceEmotions && (
            <motion.div variants={itemVariants}>
              <EmotionResults emotions={faceEmotions} inputText={"[Image Data Analysed]"} title="Face Analysis" icon={Camera} />
            </motion.div>
          )}

          {combinedEmotions && (
            <motion.div variants={itemVariants} className="col-span-full lg:col-span-1">
              <EmotionResults emotions={combinedEmotions} inputText={`Combined Modalities`} title="Overall Fusion" icon={Combine} highlight={true} />
            </motion.div>
          )}
        </motion.div>
      </motion.div>
    </div>
  );
}
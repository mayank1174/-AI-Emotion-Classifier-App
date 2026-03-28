import { useState, useEffect } from "react";
import { Mic, MicOff, Type, Sparkles, Combine, Loader2, Zap, Activity } from "lucide-react";
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

async function fetchClassifyAudio(audioBlob: Blob): Promise<EmotionScore[]> {
  const formData = new FormData();
  formData.append('audio', audioBlob, 'recording.wav');
  const res = await fetch('/api/classify_audio', {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error('Audio classification failed');
  return res.json();
}

async function fetchCombine(...sets: (EmotionScore[] | null)[]): Promise<EmotionScore[]> {
  const validSets = sets.filter(Boolean) as EmotionScore[][];
  if (validSets.length < 1) return [];
  const res = await fetch('/api/combine', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ emotions1: validSets[0], emotions2: validSets[1] || [] }),
  });
  if (!res.ok) throw new Error('Combine failed');
  return res.json();
}

export function EmotionClassifier() {
  const [textInput, setTextInput] = useState("");
  const [isRecording, setIsRecording] = useState(false);

  const [isLoadingText, setIsLoadingText] = useState(false);
  const [isLoadingSpeech, setIsLoadingSpeech] = useState(false);

  const [transcript, setTranscript] = useState("");
  const [textEmotions, setTextEmotions] = useState<EmotionScore[] | null>(null);
  const [speechEmotions, setSpeechEmotions] = useState<EmotionScore[] | null>(null);
  const [combinedEmotions, setCombinedEmotions] = useState<EmotionScore[] | null>(null);

  const [recognition, setRecognition] = useState<any>(null);
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [audioChunks, setAudioChunks] = useState<Blob[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Initialize Web Speech API for live transcript display
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      if (SpeechRecognition && !recognition) {
        const instance = new SpeechRecognition();
        instance.continuous = true;
        instance.interimResults = true;
        instance.lang = 'en-US';

        instance.onstart = () => {
          setIsRecording(true);
          setTranscript("");
          setSpeechEmotions(null);
        };

        instance.onresult = (event: any) => {
          let interim = '';
          let final = '';
          for (let i = event.resultIndex; i < event.results.length; i++) {
            const piece = event.results[i][0].transcript;
            event.results[i].isFinal ? (final += piece + ' ') : (interim += piece);
          }
          setTranscript((prev) => prev + final || interim);
        };

        instance.onerror = () => setIsRecording(false);
        instance.onend = () => setIsRecording(false);

        setRecognition(instance);
      }
    }
  }, []);

  const updateCombinations = async (t: EmotionScore[] | null, s: EmotionScore[] | null) => {
    try {
      const valid = [t, s].filter(Boolean);
      if (valid.length > 0) {
        const combined = await fetchCombine(t, s);
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
      await updateCombinations(results, speechEmotions);
    } catch {
      setError('Could not reach the backend. Make sure app.py is running.');
    } finally {
      setIsLoadingText(false);
    }
  };

  const startRecording = async () => {
    if (!recognition) {
      alert('Speech recognition not supported. Please use Chrome or Edge.');
      return;
    }
    // Start MediaRecorder to capture actual audio bytes for MFCC
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const chunks: Blob[] = [];
      const recorder = new MediaRecorder(stream);
      recorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
      recorder.onstop = async () => {
        stream.getTracks().forEach(t => t.stop());
        const blob = new Blob(chunks, { type: 'audio/webm' });
        setIsLoadingSpeech(true);
        setError(null);
        try {
          const results = await fetchClassifyAudio(blob);
          setSpeechEmotions(results);
          await updateCombinations(textEmotions, results);
        } catch {
          setError('Could not reach backend for audio analysis.');
        } finally {
          setIsLoadingSpeech(false);
        }
      };
      recorder.start();
      setMediaRecorder(recorder);
      setAudioChunks(chunks);
    } catch {
      setError('Microphone access denied or unavailable.');
      return;
    }
    recognition.start();
  };

  const stopRecording = () => {
    recognition?.stop();
    mediaRecorder?.stop();
    setIsRecording(false);
  };

  const clearAll = () => {
    setTextInput("");
    setTranscript("");
    setTextEmotions(null);
    setSpeechEmotions(null);
    setCombinedEmotions(null);
    setError(null);
    if (recognition && isRecording) recognition.stop();
    mediaRecorder?.stop();
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.18, delayChildren: 0.05 }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 22 },
    show: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.65, ease: [0.22, 1, 0.36, 1] as const }
    }
  };

  return (
    <div className="w-full relative py-12">
      {/* ── Hero / Title Section ── */}
      <motion.div
        className="flex flex-col items-center justify-center min-h-[30vh] text-center px-6 relative"
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
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
          Analyze emotions from text and speech simultaneously for deeper, comprehensive insights.
        </motion.p>
      </motion.div>

      {/* ── Main Content ── */}
      <motion.div
        className="max-w-4xl mx-auto px-6 pb-8 mt-8"
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

        {/* Input Section — 2 cards side by side */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">

          {/* 1. Text Input Card */}
          <motion.div variants={itemVariants} whileHover={{ y: -5, transition: { type: 'spring', stiffness: 320, damping: 22 } }} className="flex">
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
                  onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setTextInput(e.target.value)}
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
          <motion.div variants={itemVariants} whileHover={{ y: -5, transition: { type: 'spring', stiffness: 320, damping: 22 } }} className="flex">
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
                      "{transcript}"
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
                    disabled={isLoadingSpeech}
                    className="w-full bg-gradient-to-r from-secondary to-pink-500 hover:from-secondary/90 hover:to-pink-500/90 text-white rounded-xl shadow-lg shadow-secondary/25 h-12 text-base font-semibold"
                  >
                    {isLoadingSpeech ? (
                      <><Loader2 className="w-5 h-5 mr-2 animate-spin" /> Analyzing...</>
                    ) : (
                      <><Mic className="w-5 h-5 mr-2" /> Start Recording</>
                    )}
                  </Button>
                ) : (
                  <Button
                    onClick={stopRecording}
                    variant="outline"
                    className="w-full border-secondary/50 text-secondary hover:bg-secondary/10 hover:text-secondary rounded-xl h-12 text-base font-semibold"
                  >
                    <MicOff className="w-5 h-5 mr-2" /> Stop & Analyze
                  </Button>
                )}
              </CardContent>
            </Card>
          </motion.div>

        </div>

        {/* Clear Button */}
        <AnimatePresence>
          {(textEmotions || speechEmotions) && (
            <motion.div
              initial={{ opacity: 0, scale: 0.92, y: 8 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.92, y: 8 }}
              transition={{ type: 'spring', stiffness: 280, damping: 24 }}
              className="flex justify-center mb-10"
            >
              <Button onClick={clearAll} variant="outline" className="rounded-full px-8 shadow-sm">
                Clear Results
              </Button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Results Section */}
        <motion.div variants={containerVariants} className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6 mb-2">
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

          {combinedEmotions && (
            <motion.div variants={itemVariants} className="col-span-full lg:col-span-1">
              <EmotionResults emotions={combinedEmotions} inputText="Combined Modalities" title="Overall Fusion" icon={Combine} highlight={true} />
            </motion.div>
          )}
        </motion.div>
      </motion.div>
    </div>
  );
}
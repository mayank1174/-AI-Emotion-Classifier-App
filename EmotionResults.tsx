import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { EmotionScore } from "./EmotionClassifier";
import { Smile, Frown, Angry, AlertTriangle, Zap, Minus, LucideIcon, Heart, Flame, Cloud, HelpCircle, Sparkles, Droplet, Wind } from "lucide-react";
import { motion } from "framer-motion";

interface EmotionResultsProps {
  emotions: EmotionScore[];
  inputText: string;
  title?: string;
  icon?: LucideIcon;
  highlight?: boolean;
}

const emotionIcons: Record<string, any> = {
  'Happy': Smile,
  'Sad': Frown,
  'Angry': Angry,
  'Fearful': AlertTriangle,
  'Anxious': Heart,
  'Stressed': Flame,
  'Worried': Cloud,
  'Confused': HelpCircle,
  'Excited': Sparkles,
  'Surprised': Zap,
  'Disgusted': Droplet,
  'Calm': Wind,
  'Neutral': Minus
};

export function EmotionResults({ emotions, inputText, title = "Emotion Analysis Results", icon: HeaderIcon, highlight = false }: EmotionResultsProps) {
  const dominantEmotion = emotions[0];
  const Icon = emotionIcons[dominantEmotion.emotion] || Smile;

  return (
    <Card className={`h-full border-border/50 backdrop-blur-xl bg-card transition-all ${
      highlight 
        ? 'shadow-xl shadow-purple-500/10 ring-1 ring-primary/30 dark:ring-primary/40 bg-gradient-to-br from-primary/5 to-secondary/5' 
        : 'shadow-xl shadow-primary/5'
    }`}>
      <CardHeader className="pb-3 border-b border-border/40">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-lg font-bold">
            {HeaderIcon && <HeaderIcon className={`w-5 h-5 ${highlight ? 'text-primary' : 'text-muted-foreground'}`} />}
            {title}
          </CardTitle>
          {highlight && (
            <Badge className="bg-gradient-to-r from-primary to-secondary text-white border-0 shadow-sm">
              <Sparkles className="w-3 h-3 mr-1" /> Fusion Insight
            </Badge>
          )}
        </div>
      </CardHeader>
      
      <CardContent className="space-y-6 pt-5">
        {/* Dominant Emotion */}
        <div className={`p-5 rounded-2xl border ${
          highlight 
            ? 'bg-gradient-to-r from-primary/10 to-secondary/10 border-primary/20' 
            : 'bg-muted/30 border-border/50'
        }`}>
          <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3">Dominant Emotion</p>
          <div className="flex items-center gap-4">
            <div 
              className="w-14 h-14 rounded-2xl flex items-center justify-center shadow-sm"
              style={{ backgroundColor: `${dominantEmotion.color}15`, border: `1px solid ${dominantEmotion.color}30` }}
            >
              <Icon className="w-7 h-7" style={{ color: dominantEmotion.color }} />
            </div>
            <div>
              <h3 className="text-3xl font-black mb-1 drop-shadow-sm" style={{ color: dominantEmotion.color }}>
                {dominantEmotion.emotion}
              </h3>
              <Badge variant="outline" className="font-medium bg-background/50 border-border/50">
                {dominantEmotion.score.toFixed(1)}% confidence
              </Badge>
            </div>
          </div>
        </div>

        {/* All Emotions List */}
        <div>
          <h4 className="text-sm font-semibold mb-3 text-foreground/80 flex items-center gap-2">
            Detailed Breakdown
          </h4>
          <div className="space-y-3">
            {emotions.map((emotion, index) => {
              const EmotionIcon = emotionIcons[emotion.emotion] || Smile;
              return (
                <div key={emotion.emotion} className="flex items-center gap-3">
                  <div className="w-6 h-6 rounded-md flex items-center justify-center bg-muted/50 border border-border/30">
                    <EmotionIcon className="w-3.5 h-3.5" style={{ color: emotion.color }} />
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="text-sm font-medium text-foreground/90">{emotion.emotion}</span>
                      <span className="text-xs font-semibold text-muted-foreground">{emotion.score.toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-muted rounded-full h-2.5 overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${emotion.score}%` }}
                        transition={{
                          type: 'spring',
                          stiffness: 60,
                          damping: 18,
                          delay: 0.08 + index * 0.04
                        }}
                        className="h-full rounded-full"
                        style={{ backgroundColor: emotion.color }}
                      />
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Bar Chart */}
        <div className="pt-2">
          <h4 className="text-sm font-semibold mb-3 text-foreground/80">Score Distribution</h4>
          <div className="bg-card rounded-xl border border-border/40 p-2 shadow-sm">
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={emotions} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" opacity={0.5} />
                <XAxis 
                  dataKey="emotion" 
                  tick={{ fontSize: 11, fill: 'var(--muted-foreground)' }}
                  axisLine={false}
                  tickLine={false}
                  angle={-45}
                  textAnchor="end"
                  height={60}
                />
                <YAxis 
                  tick={{ fontSize: 11, fill: 'var(--muted-foreground)' }} 
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip 
                  cursor={{ fill: 'var(--muted)', opacity: 0.4 }}
                  contentStyle={{ 
                    borderRadius: '12px', 
                    border: '1px solid var(--border)',
                    backgroundColor: 'var(--card)',
                    boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'
                  }}
                  itemStyle={{ fontWeight: 600 }}
                  formatter={(value: any) => `${Number(value).toFixed(1)}%`}
                />
                <Bar dataKey="score" radius={[4, 4, 0, 0]} maxBarSize={40}>
                  {emotions.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Input Preview */}
        <div className="p-4 bg-muted/40 rounded-xl border border-border/40 relative overflow-hidden">
          <div className="absolute top-0 left-0 w-1 h-full bg-gradient-to-b from-primary/50 to-secondary/50" />
          <p className="text-[11px] font-bold uppercase text-muted-foreground mb-1.5 tracking-wider">Source Data</p>
          <p className="text-sm text-foreground/80 line-clamp-3 whitespace-pre-wrap leading-relaxed font-medium">
            "{inputText}"
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
import { ThemeProvider } from "next-themes";
import { EmotionClassifier } from "./components/EmotionClassifier";
import { Navbar } from "./components/ui/Navbar";
import { Footer } from "./components/ui/Footer";

export default function App() {
  return (
    <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
      <div className="relative min-h-screen flex flex-col bg-background font-sans antialiased text-foreground selection:bg-purple-500/30 overflow-x-hidden">
        <Navbar />
        <main className="flex-1 relative">
          {/* Very subtle 3-color gradient backgrounds (Purple -> Base -> Pink) fixed to the viewport to prevent scroll lag/sliders */}
          <div className="fixed inset-0 z-[-1] pointer-events-none">
            <div className="absolute inset-0 bg-gradient-to-br from-purple-100/50 via-background to-pink-100/50 dark:hidden" />
            <div className="absolute inset-0 hidden dark:block bg-gradient-to-br from-purple-900/20 via-background to-pink-900/20" />
          </div>
          
          <EmotionClassifier />
        </main>
        <Footer />
      </div>
    </ThemeProvider>
  );
}
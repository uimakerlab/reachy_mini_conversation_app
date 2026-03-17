import { useState, useEffect, useCallback, useRef } from "react";
import { Box } from "@mui/material";

const REACHIES = [
  "/avatars/bored-teenager.svg",
  "/avatars/captain-circuit.svg",
  "/avatars/chess-coach.svg",
  "/avatars/cosmic-kitchen.svg",
  "/avatars/hype-bot.svg",
  "/avatars/mad-scientist.svg",
  "/avatars/mars-rover.svg",
  "/avatars/nature-doc.svg",
  "/avatars/noir-detective.svg",
  "/avatars/sorry-bro.svg",
  "/avatars/time-traveler.svg",
  "/avatars/victorian-butler.svg",
];

interface Props {
  width?: number;
  height?: number;
  interval?: number;
  fadeInDuration?: number;
  fadeOutDuration?: number;
  zoom?: number;
}

export default function ReachiesCarousel({
  width = 120,
  height = 120,
  interval = 600,
  fadeInDuration = 200,
  fadeOutDuration = 80,
  zoom = 1,
}: Props) {
  const [ready, setReady] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [previousIndex, setPreviousIndex] = useState<number | null>(null);
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [fadeOutComplete, setFadeOutComplete] = useState(false);
  const currentIndexRef = useRef(currentIndex);

  useEffect(() => {
    currentIndexRef.current = currentIndex;
  }, [currentIndex]);

  useEffect(() => {
    let loaded = 0;
    REACHIES.forEach((src) => {
      const img = new Image();
      img.onload = img.onerror = () => {
        loaded++;
        if (loaded >= REACHIES.length) setReady(true);
      };
      img.src = src;
    });
  }, []);

  const getRandomIndex = useCallback((cur: number, total: number) => {
    if (total <= 1) return 0;
    let next: number;
    do {
      next = Math.floor(Math.random() * total);
    } while (next === cur);
    return next;
  }, []);

  useEffect(() => {
    if (!ready) return;

    const pendingTimers: ReturnType<typeof setTimeout>[] = [];

    const timer = setInterval(() => {
      const prevIdx = currentIndexRef.current;
      setPreviousIndex(prevIdx);
      setIsTransitioning(true);
      setFadeOutComplete(false);

      const newIndex = getRandomIndex(prevIdx, REACHIES.length);
      setCurrentIndex(newIndex);

      const overlapDelay = Math.min(fadeInDuration * 0.4, fadeOutDuration * 2);
      pendingTimers.push(setTimeout(() => setFadeOutComplete(true), overlapDelay));

      pendingTimers.push(setTimeout(() => {
        setIsTransitioning(false);
        setPreviousIndex(null);
        setFadeOutComplete(false);
      }, Math.max(fadeInDuration, fadeOutDuration)));
    }, interval);

    return () => {
      clearInterval(timer);
      pendingTimers.forEach(clearTimeout);
    };
  }, [ready, interval, fadeInDuration, fadeOutDuration, getRandomIndex]);

  if (!ready) {
    return <Box sx={{ width, height }} />;
  }

  return (
    <Box
      sx={{
        position: "relative",
        width,
        height,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        overflow: "hidden",
        mx: "auto",
      }}
    >
      {REACHIES.map((src, index) => {
        const isActive = index === currentIndex;
        const isPrevious = index === previousIndex && isTransitioning;

        let opacity = 0;
        let transition = "none";

        if (isActive) {
          opacity = 1;
          transition = `opacity ${fadeInDuration}ms cubic-bezier(0.4, 0, 0.2, 1)`;
        } else if (isPrevious) {
          opacity = fadeOutComplete ? 0 : 1;
          transition = `opacity ${fadeOutDuration}ms cubic-bezier(0.4, 0, 1, 1)`;
        }

        return (
          <Box
            key={src}
            component="img"
            src={src}
            alt={`Reachy ${index + 1}`}
            sx={{
              position: "absolute",
              width: width * zoom,
              height: height * zoom,
              objectFit: "contain",
              opacity,
              transform: "translate(-50%, -50%)",
              transition,
              pointerEvents: "none",
              left: "50%",
              top: "50%",
              zIndex: isActive ? 2 : isPrevious ? 1 : 0,
              willChange: "opacity",
            }}
          />
        );
      })}
    </Box>
  );
}

import { useEffect, useRef, useState } from 'react';

const CLASS_COLORS = {
  'Bacterial Spot': 'other-bar',
  'Early Blight': 'other-bar',
  'Late Blight': 'other-bar',
  'Leaf Mold': 'other-bar',
  'Septoria Leaf Spot': 'other-bar',
  'Spider Mites': 'other-bar',
  'Target Spot': 'other-bar',
  'Yellow Leaf Curl Virus': 'other-bar',
  'Mosaic Virus': 'other-bar',
  'Healthy': 'healthy-bar',
};

export default function ProbabilityChart({ probabilities, topClass }) {
  const [animated, setAnimated] = useState(false);
  const chartRef = useRef(null);

  useEffect(() => {
    setAnimated(false);
    const timer = setTimeout(() => setAnimated(true), 100);
    return () => clearTimeout(timer);
  }, [probabilities]);

  if (!probabilities) return null;

  const sorted = Object.entries(probabilities).sort(([, a], [, b]) => b - a);

  return (
    <div className="chart-bars" ref={chartRef}>
      {sorted.map(([label, prob], i) => {
        const isTop = label === topClass;
        const pct = (prob * 100).toFixed(1);
        const barClass = isTop && label !== 'Healthy'
          ? 'top-bar'
          : CLASS_COLORS[label] || 'other-bar';

        return (
          <div
            key={label}
            className="chart-bar-row"
            style={{ animationDelay: `${i * 0.06}s` }}
          >
            <span className={`chart-bar-label ${isTop ? 'is-top' : ''}`}>
              {label}
            </span>
            <div className="chart-bar-track">
              <div
                className={`chart-bar-fill ${barClass}`}
                style={{ width: animated ? `${Math.max(prob * 100, 0.5)}%` : '0%' }}
              />
            </div>
            <span className={`chart-bar-pct ${isTop ? 'is-top' : ''}`}>
              {pct}%
            </span>
          </div>
        );
      })}
    </div>
  );
}

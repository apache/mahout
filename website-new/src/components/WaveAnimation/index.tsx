/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {useEffect, useRef} from 'react';
import styles from './styles.module.css';

interface Wave {
  timeModifier: number;
  lineWidth: number;
  amplitude: number;
  wavelength: number;
  segmentLength?: number;
  strokeStyle?: string | CanvasGradient;
}

export default function WaveAnimation(): JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const timeRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const waves: Wave[] = [
      {
        timeModifier: 1,
        lineWidth: 3,
        amplitude: 150,
        wavelength: 200,
        segmentLength: 20,
      },
      {
        timeModifier: 1,
        lineWidth: 2,
        amplitude: 250,
        wavelength: 200,
      },
      {
        timeModifier: 1,
        lineWidth: 3,
        amplitude: -150,
        wavelength: 50,
        segmentLength: 10,
      },
      {
        timeModifier: 1,
        lineWidth: 1,
        amplitude: -100,
        wavelength: 100,
        segmentLength: 20,
      },
    ];

    const speed = 8;
    const defaultSegmentLength = 10;
    const PI2 = Math.PI * 2;
    const HALFPI = Math.PI / 2;

    let width: number;
    let height: number;
    let waveWidth: number;
    let waveLeft: number;
    let dpr: number;

    const ease = (percent: number, amplitude: number): number => {
      return amplitude * (Math.sin(percent * PI2 - HALFPI) + 1) * 0.5;
    };

    const resizeCanvas = () => {
      dpr = window.devicePixelRatio || 1;
      width = canvas.width = document.body.clientWidth * dpr;
      height = canvas.height = 500 * dpr;
      canvas.style.width = document.body.clientWidth + 'px';
      canvas.style.height = '500px';
      waveWidth = width * 0.95;
      waveLeft = width * 0.25;

      // Create gradient
      const gradient = ctx.createLinearGradient(0, 0, width, 0);
      gradient.addColorStop(0, 'rgba(254, 255, 255, 0)');
      gradient.addColorStop(0.5, 'rgba(255, 255, 255, 0.5)');
      gradient.addColorStop(1, 'rgba(255, 255, 254, 0)');

      waves.forEach((wave) => {
        wave.strokeStyle = gradient;
      });
    };

    const drawSine = (time: number, wave: Wave) => {
      const amplitude = wave.amplitude;
      const wavelength = wave.wavelength;
      const lineWidth = wave.lineWidth;
      const strokeStyle = wave.strokeStyle || 'rgba(255, 255, 255, 0.2)';
      const segmentLength = wave.segmentLength || defaultSegmentLength;

      const yAxis = height / 2;

      ctx.lineWidth = lineWidth * dpr;
      ctx.strokeStyle = strokeStyle;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.beginPath();

      ctx.moveTo(0, yAxis);
      ctx.lineTo(waveLeft, yAxis);

      for (let i = 0; i < waveWidth; i += segmentLength) {
        const x = time * speed + (-yAxis + i) / wavelength;
        const y = Math.sin(x);
        const amp = ease(i / waveWidth, amplitude);
        ctx.lineTo(i + waveLeft, amp * y + yAxis);
      }

      ctx.lineTo(width, yAxis);
      ctx.stroke();
    };

    const loop = () => {
      timeRef.current -= 0.007;
      ctx.clearRect(0, 0, width, height);

      waves.forEach((wave) => {
        const timeModifier = wave.timeModifier || 1;
        drawSine(timeRef.current * timeModifier, wave);
      });

      animationRef.current = requestAnimationFrame(loop);
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    loop();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  return <canvas ref={canvasRef} className={styles.waveCanvas} />;
}

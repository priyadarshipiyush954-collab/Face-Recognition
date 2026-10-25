/**
 * AURA Waterflow Engine
 * Interactive multi-layered organic fluid wave simulation, bioluminescent micro-bubbles,
 * mouse-reactive ripple physics, shockwave clicks, and scroll velocity dynamics.
 */

class WaterflowCanvas {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext('2d');
        this.width = window.innerWidth;
        this.height = window.innerHeight;
        this.pixelRatio = Math.min(window.devicePixelRatio || 1, 2);

        this.mouse = { x: this.width / 2, y: this.height / 2, targetX: this.width / 2, targetY: this.height / 2, speed: 0 };
        this.lastMouse = { x: this.width / 2, y: this.height / 2 };
        this.ripples = [];
        this.droplets = [];
        this.scrollVelocity = 0;
        this.lastScrollY = window.scrollY;
        this.time = 0;

        // Wave Layer Configurations (Cascading fluid ribbons)
        this.waves = [
            { amplitude: 45, wavelength: 0.0035, speed: 0.018, offset: 0, color: 'rgba(0, 242, 254, 0.07)', yRatio: 0.35 },
            { amplitude: 65, wavelength: 0.0022, speed: 0.012, offset: 2.1, color: 'rgba(79, 172, 254, 0.06)', yRatio: 0.55 },
            { amplitude: 85, wavelength: 0.0018, speed: 0.015, offset: 4.2, color: 'rgba(121, 40, 202, 0.05)', yRatio: 0.75 },
            { amplitude: 50, wavelength: 0.0040, speed: 0.022, offset: 1.2, color: 'rgba(0, 245, 160, 0.04)', yRatio: 0.88 },
        ];

        this.init();
    }

    init() {
        this.resize();
        this.createDroplets(45);
        this.bindEvents();
        this.animate();
    }

    resize() {
        this.width = window.innerWidth;
        this.height = window.innerHeight;
        this.canvas.width = this.width * this.pixelRatio;
        this.canvas.height = this.height * this.pixelRatio;
        this.canvas.style.width = `${this.width}px`;
        this.canvas.style.height = `${this.height}px`;
        this.ctx.scale(this.pixelRatio, this.pixelRatio);
    }

    createDroplets(count) {
        for (let i = 0; i < count; i++) {
            this.droplets.push({
                x: Math.random() * this.width,
                y: Math.random() * this.height,
                radius: Math.random() * 2.5 + 0.8,
                speedY: -(Math.random() * 0.6 + 0.2),
                speedX: (Math.random() - 0.5) * 0.3,
                baseAlpha: Math.random() * 0.5 + 0.2,
                glow: Math.random() > 0.5 ? '#00f2fe' : '#4facfe',
                phase: Math.random() * Math.PI * 2,
            });
        }
    }

    bindEvents() {
        window.addEventListener('resize', () => this.resize());

        window.addEventListener('mousemove', (e) => {
            this.mouse.targetX = e.clientX;
            this.mouse.targetY = e.clientY;

            // Calculate mouse velocity for fluid wake
            const dx = e.clientX - this.lastMouse.x;
            const dy = e.clientY - this.lastMouse.y;
            this.mouse.speed = Math.min(Math.sqrt(dx * dx + dy * dy), 50);
            this.lastMouse.x = e.clientX;
            this.lastMouse.y = e.clientY;

            if (this.mouse.speed > 12 && Math.random() > 0.6) {
                this.addRipple(e.clientX, e.clientY, 0.3 + this.mouse.speed / 70);
            }
        });

        window.addEventListener('click', (e) => {
            this.addRipple(e.clientX, e.clientY, 1.2);
            this.createSplash(e.clientX, e.clientY);
        });

        window.addEventListener('scroll', () => {
            const currentY = window.scrollY;
            const delta = currentY - this.lastScrollY;
            this.scrollVelocity = delta * 0.15;
            this.lastScrollY = currentY;

            // Update waterflow scroll progress indicator
            this.updateScrollIndicator();
        }, { passive: true });
    }

    addRipple(x, y, strength = 1.0) {
        if (this.ripples.length > 25) this.ripples.shift();
        this.ripples.push({
            x,
            y,
            radius: 5,
            maxRadius: 180 * strength,
            alpha: 0.65 * strength,
            speed: 3.5 * strength,
            color: strength > 0.8 ? 'rgba(0, 242, 254,' : 'rgba(79, 172, 254,',
        });
    }

    createSplash(x, y) {
        for (let i = 0; i < 8; i++) {
            const angle = Math.random() * Math.PI * 2;
            const speed = Math.random() * 3 + 1;
            this.droplets.push({
                x,
                y,
                radius: Math.random() * 2 + 1,
                speedX: Math.cos(angle) * speed,
                speedY: Math.sin(angle) * speed - 1.5,
                baseAlpha: 0.8,
                glow: '#00f5a0',
                phase: 0,
                isSplash: true,
                life: 60,
            });
        }
    }

    updateScrollIndicator() {
        const indicator = document.getElementById('waterflow-stream-bar');
        if (indicator) {
            const docHeight = document.documentElement.scrollHeight - window.innerHeight;
            const progress = docHeight > 0 ? (window.scrollY / docHeight) * 100 : 0;
            indicator.style.height = `${progress}%`;
        }
    }

    animate() {
        this.time += 0.02 + Math.abs(this.scrollVelocity) * 0.005;
        this.scrollVelocity *= 0.92; // smooth damping

        // Lerp mouse
        this.mouse.x += (this.mouse.targetX - this.mouse.x) * 0.08;
        this.mouse.y += (this.mouse.targetY - this.mouse.y) * 0.08;

        this.ctx.clearRect(0, 0, this.width, this.height);

        // 1. Draw flowing liquid wave ribbons
        this.drawWaves();

        // 2. Draw expanding water ripples
        this.drawRipples();

        // 3. Draw floating bioluminescent micro-droplets
        this.drawDroplets();

        requestAnimationFrame(() => this.animate());
    }

    drawWaves() {
        this.waves.forEach((w) => {
            this.ctx.save();
            this.ctx.beginPath();

            const baseY = this.height * w.yRatio;
            this.ctx.moveTo(0, this.height);
            this.ctx.lineTo(0, baseY);

            // Calculate sinusoidal curve with mouse & scroll disturbances
            for (let x = 0; x <= this.width; x += 15) {
                const distToMouse = Math.abs(x - this.mouse.x);
                const mouseInfluence = Math.max(0, 1 - distToMouse / 350) * Math.sin(this.time * 2) * 25;

                const y = baseY +
                    Math.sin(x * w.wavelength + this.time * w.speed * 60 + w.offset) * w.amplitude +
                    Math.cos(x * (w.wavelength * 0.6) + this.time * 0.5) * (w.amplitude * 0.4) +
                    mouseInfluence;

                this.ctx.lineTo(x, y);
            }

            this.ctx.lineTo(this.width, this.height);
            this.ctx.closePath();

            this.ctx.fillStyle = w.color;
            this.ctx.fill();
            this.ctx.restore();
        });
    }

    drawRipples() {
        for (let i = this.ripples.length - 1; i >= 0; i--) {
            const r = this.ripples[i];
            r.radius += r.speed;
            r.alpha *= 0.96;

            if (r.alpha < 0.01 || r.radius >= r.maxRadius) {
                this.ripples.splice(i, 1);
                continue;
            }

            this.ctx.save();
            this.ctx.beginPath();
            this.ctx.arc(r.x, r.y, r.radius, 0, Math.PI * 2);
            this.ctx.strokeStyle = `${r.color} ${r.alpha})`;
            this.ctx.lineWidth = 1.5;
            this.ctx.stroke();

            // Inner subtle secondary echo ring
            if (r.radius > 20) {
                this.ctx.beginPath();
                this.ctx.arc(r.x, r.y, r.radius * 0.65, 0, Math.PI * 2);
                this.ctx.strokeStyle = `${r.color} ${r.alpha * 0.4})`;
                this.ctx.lineWidth = 1;
                this.ctx.stroke();
            }
            this.ctx.restore();
        }
    }

    drawDroplets() {
        for (let i = this.droplets.length - 1; i >= 0; i--) {
            const d = this.droplets[i];

            if (d.isSplash) {
                d.x += d.speedX;
                d.y += d.speedY;
                d.speedY += 0.12; // gravity
                d.life -= 1;
                d.baseAlpha *= 0.96;
                if (d.life <= 0) {
                    this.droplets.splice(i, 1);
                    continue;
                }
            } else {
                d.y += d.speedY - (this.scrollVelocity * 0.2);
                d.x += d.speedX + Math.sin(this.time + d.phase) * 0.4;

                // Wrap around edges
                if (d.y < -10) {
                    d.y = this.height + 10;
                    d.x = Math.random() * this.width;
                }
                if (d.y > this.height + 10) d.y = -10;
                if (d.x < -10) d.x = this.width + 10;
                if (d.x > this.width + 10) d.x = -10;
            }

            const currentAlpha = d.baseAlpha * (0.8 + Math.sin(this.time * 2 + d.phase) * 0.2);

            this.ctx.save();
            this.ctx.beginPath();
            this.ctx.arc(d.x, d.y, d.radius, 0, Math.PI * 2);
            this.ctx.fillStyle = d.glow;
            this.ctx.globalAlpha = Math.max(0, currentAlpha);
            this.ctx.shadowBlur = 8;
            this.ctx.shadowColor = d.glow;
            this.ctx.fill();
            this.ctx.restore();
        }
    }
}

// Instantiate on DOM load
window.addEventListener('DOMContentLoaded', () => {
    window.waterflowSim = new WaterflowCanvas('waterflow-canvas');
});

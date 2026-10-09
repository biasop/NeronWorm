"""NeuroWorm entry point: graphical play or deterministic headless runs."""
import argparse
from dataclasses import replace
import logging
from math import isfinite
import os
from pathlib import Path

from config import Config


def positive_float(text: str) -> float:
    value = float(text)
    if not isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A connectome-inspired neural worm simulation")
    parser.add_argument("--headless", action="store_true", help="render using SDL dummy; run without wall-clock delay")
    parser.add_argument("--seconds", type=positive_float, help="stop after this many simulated seconds (headless default: 10)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fps", type=int, default=60, help="render FPS; does not change physics/neural dt")
    parser.add_argument("--speed", type=positive_float, default=1.0)
    parser.add_argument("--connectome", type=Path, help="JSON connectome with sensor/motor mapping metadata")
    parser.add_argument("--results", type=Path, help="episode CSV output")
    parser.add_argument("--screenshot", type=Path, help="save final rendered frame as PNG")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"
        if args.seconds is None:
            args.seconds = 10.0
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    # Set SDL environment before importing/initializing Pygame.
    import pygame
    from simulation import Simulation
    from visualization.brain_view import BrainView
    from visualization.hud import HUD

    base = Config()
    config = replace(base, seed=args.seed, render_fps=args.fps,
                     connectome_path=args.connectome or base.connectome_path,
                     results_path=args.results or base.results_path)
    pygame.display.init()
    pygame.font.init()
    sim: Simulation | None = None
    try:
        screen = pygame.display.set_mode((config.world_width + config.brain_panel_width, config.world_height))
        pygame.display.set_caption("NeuroWorm · neural survival sandbox")
        clock = pygame.time.Clock()
        sim = Simulation(config)
        brain_view = BrainView(sim.connectome, config.brain_panel_width, config.world_height)
        hud = HUD()
        running, paused, show_brain, debug = True, False, True, args.debug
        speed = min(8.0, max(0.25, args.speed))
        advanced_s = 0.0
        logging.info("Started seed=%s, SDL=%s, neural dt=%gms, body dt=%gs", config.seed,
                     pygame.display.get_driver(), config.brain_dt_ms, config.physics_dt_s)
        next_log = 5.0
        while running:
            elapsed_s = 1.0 / config.render_fps if args.headless else min(clock.tick(config.render_fps) / 1000.0, 0.1)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        sim.reset_episode()
                    elif event.key == pygame.K_b:
                        show_brain = not show_brain
                    elif event.key == pygame.K_d:
                        debug = not debug
                    elif event.key == pygame.K_UP:
                        speed = min(8.0, speed * 2)
                    elif event.key == pygame.K_DOWN:
                        speed = max(0.25, speed / 2)
            if not paused and running:
                duration = elapsed_s * speed
                if args.seconds is not None:
                    # Account for accumulated fractions of a physics tick too.
                    duration = min(duration, max(0.0, args.seconds - advanced_s))
                sim.advance(duration)
                advanced_s += duration
            world_surface = screen.subsurface((0, 0, config.world_width, config.world_height))
            sim.world.draw(world_surface, debug)
            sim.worm.draw(world_surface)
            hud.draw(world_surface, sim, paused, speed, debug, clock.get_fps())
            panel = screen.subsurface((config.world_width, 0, config.brain_panel_width, config.world_height))
            brain_view.draw(panel, sim.brain, show_brain)
            pygame.display.flip()
            if sim.total_time_s + 1e-9 >= next_log:
                logging.info("t=%.1fs episode=%s xy=(%.1f, %.1f) food=%s health=%.1f energy=%.1f motors=(%.3f, %.3f)",
                             sim.total_time_s, sim.episode, sim.worm.x, sim.worm.y, sim.food_eaten,
                             sim.worm.health, sim.worm.energy, sim.worm.muscles.left, sim.worm.muscles.right)
                next_log += 5.0
            if args.seconds is not None and advanced_s + 1e-9 >= args.seconds:
                running = False
        if args.screenshot is not None:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(screen, str(args.screenshot))
            logging.info("Saved screenshot: %s", args.screenshot)
        return 0
    finally:
        if sim is not None:
            sim.close()
        pygame.quit()


if __name__ == "__main__":
    raise SystemExit(main())

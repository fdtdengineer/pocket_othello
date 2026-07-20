#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import { once } from 'node:events';
import {
  BLACK,
  applyMove,
  createInitialBoard,
  getLegalMoves,
  getNextTurn,
} from './rules.js';
import { chooseHardMove } from './search.js';

const DEFAULTS = {
  games: 100,
  output: 'engine/python/data/teacher.jsonl',
  seed: 20260720,
  timeMs: 100,
  maxDepth: 6,
  exploration: 0.20,
  maxExamples: Number.POSITIVE_INFINITY,
};

function printHelp() {
  console.log(`Generate Hard-CPU teacher labels as JSONL.

Usage:
  node engine/javascript/generate_teacher_data.mjs [options]

Options:
  --games <n>          Number of games to generate (default: ${DEFAULTS.games})
  --output <path>      JSONL output path (default: ${DEFAULTS.output})
  --seed <n>           Deterministic random seed (default: ${DEFAULTS.seed})
  --time-ms <n>        Hard-CPU search budget, clamped to 20-120 ms (default: ${DEFAULTS.timeMs})
  --max-depth <n>      Maximum alpha-beta depth (default: ${DEFAULTS.maxDepth})
  --exploration <x>    Probability of playing a random legal move after labeling (default: ${DEFAULTS.exploration})
  --max-examples <n>   Stop after this many labeled positions
  --help               Show this message
`);
}

function parseInteger(value, name, minimum = 0) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isInteger(parsed) || parsed < minimum) {
    throw new Error(`${name} must be an integer >= ${minimum}.`);
  }
  return parsed;
}

function parseProbability(value, name) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0 || parsed > 1) {
    throw new Error(`${name} must be between 0 and 1.`);
  }
  return parsed;
}

function parseArgs(argv) {
  const options = { ...DEFAULTS };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === '--help') {
      options.help = true;
      continue;
    }
    const value = argv[index + 1];
    if (value === undefined) throw new Error(`Missing value for ${argument}.`);
    index += 1;

    if (argument === '--games') options.games = parseInteger(value, argument, 1);
    else if (argument === '--output') options.output = value;
    else if (argument === '--seed') options.seed = parseInteger(value, argument, 0);
    else if (argument === '--time-ms') options.timeMs = parseInteger(value, argument, 1);
    else if (argument === '--max-depth') options.maxDepth = parseInteger(value, argument, 1);
    else if (argument === '--exploration') options.exploration = parseProbability(value, argument);
    else if (argument === '--max-examples') options.maxExamples = parseInteger(value, argument, 1);
    else throw new Error(`Unknown option: ${argument}`);
  }
  return options;
}

function mulberry32(seed) {
  let value = seed >>> 0;
  return () => {
    value += 0x6D2B79F5;
    let result = value;
    result = Math.imul(result ^ (result >>> 15), result | 1);
    result ^= result + Math.imul(result ^ (result >>> 7), result | 61);
    return ((result ^ (result >>> 14)) >>> 0) / 4294967296;
  };
}

function randomItem(items, random) {
  return items[Math.floor(random() * items.length)];
}

function legalMask(legalMoves) {
  const mask = Array(64).fill(false);
  for (const move of legalMoves) mask[move.index] = true;
  return mask;
}

async function writeJsonLine(stream, value) {
  if (!stream.write(`${JSON.stringify(value)}\n`)) await once(stream, 'drain');
}

async function closeStream(stream) {
  stream.end();
  await once(stream, 'finish');
}

async function generate(options) {
  const outputPath = path.resolve(options.output);
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  const stream = fs.createWriteStream(outputPath, { encoding: 'utf8' });
  const playRandom = mulberry32(options.seed);
  const teacherRandom = mulberry32(options.seed ^ 0x9E3779B9);

  let examples = 0;
  let completedGames = 0;
  let passCount = 0;
  let randomMoves = 0;

  try {
    for (let game = 0; game < options.games && examples < options.maxExamples; game += 1) {
      let board = createInitialBoard();
      let player = BLACK;

      for (let ply = 0; ply < 60 && examples < options.maxExamples; ply += 1) {
        const legalMoves = getLegalMoves(board, player);
        if (legalMoves.length === 0) break;

        const teacherMove = chooseHardMove(board, player, {
          timeLimitMs: options.timeMs,
          maxDepth: options.maxDepth,
          random: teacherRandom,
        });
        if (!teacherMove || !legalMoves.some((move) => move.index === teacherMove.index)) {
          throw new Error(`Teacher returned an illegal move in game ${game}, ply ${ply}.`);
        }

        await writeJsonLine(stream, {
          schemaVersion: 1,
          board: board.slice(),
          player,
          action: teacherMove.index,
          legalMask: legalMask(legalMoves),
          game,
          ply,
        });
        examples += 1;

        let playedMove = teacherMove;
        if (legalMoves.length > 1 && playRandom() < options.exploration) {
          playedMove = randomItem(legalMoves, playRandom);
          randomMoves += 1;
        }

        board = applyMove(board, playedMove, player);
        const turn = getNextTurn(board, player);
        if (turn.passedPlayer !== null) passCount += 1;
        if (turn.gameOver) {
          completedGames += 1;
          break;
        }
        player = turn.currentPlayer;
      }
    }
  } finally {
    await closeStream(stream);
  }

  const metadata = {
    schemaVersion: 1,
    generator: 'engine/javascript/generate_teacher_data.mjs',
    teacher: 'chooseHardMove',
    config: {
      games: options.games,
      seed: options.seed,
      timeMs: options.timeMs,
      maxDepth: options.maxDepth,
      exploration: options.exploration,
      maxExamples: Number.isFinite(options.maxExamples) ? options.maxExamples : null,
    },
    examples,
    completedGames,
    passCount,
    randomMoves,
  };
  fs.writeFileSync(`${outputPath}.meta.json`, `${JSON.stringify(metadata, null, 2)}\n`, 'utf8');
  console.error(`Wrote ${examples} teacher examples to ${outputPath}`);
  return metadata;
}

try {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printHelp();
  } else {
    await generate(options);
  }
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
}

<template>
  <div class="flex flex-col items-center gap-2">
    <div class="text-[10px] text-gray-400">
      Adjust the matrix and vector to see the transformation
    </div>

    <div class="flex gap-6 flex-wrap justify-center items-start">
      <!-- Matrix input -->
      <div class="flex flex-col gap-1">
        <div class="text-[10px] font-bold text-gray-300">Matrix A</div>
        <div class="grid grid-cols-2 gap-0.5">
          <div
            v-for="(val, idx) in matrix.flat()"
            :key="idx"
            class="flex items-center"
          >
            <input
              type="number"
              :value="val"
              @input="updateMatrix(idx, $event)"
              class="w-10 px-0.5 py-0.5 bg-gray-700 border border-gray-600 rounded text-center text-white text-[10px]"
              step="0.1"
            />
          </div>
        </div>
      </div>

      <!-- Preset buttons -->
      <div class="flex flex-col gap-1">
        <div class="text-[10px] font-bold text-gray-300">Presets</div>
        <div class="flex gap-1 flex-wrap max-w-[140px]">
          <button
            v-for="preset in presets"
            :key="preset.name"
            @click="applyPreset(preset)"
            class="px-1.5 py-0.5 text-[9px] rounded bg-gray-700 hover:bg-gray-600 text-gray-300"
          >
            {{ preset.name }}
          </button>
        </div>
      </div>
    </div>

    <!-- SVG visualization -->
    <svg
      :width="svgSize"
      :height="svgSize"
      class="border border-gray-700 rounded"
      style="margin: 0"
    >
      <!-- Grid lines -->
      <g v-for="i in gridLines" :key="'grid-' + i">
        <line
          :x1="center.x + (-gridRange) * scale"
          :y1="center.y - i * scale"
          :x2="center.x + gridRange * scale"
          :y2="center.y - i * scale"
          stroke="#374151"
          stroke-width="0.5"
        />
        <line
          :x1="center.x + i * scale"
          :y1="center.y - (-gridRange) * scale"
          :x2="center.x + i * scale"
          :y2="center.y - gridRange * scale"
          stroke="#374151"
          stroke-width="0.5"
        />
      </g>

      <!-- Original axes -->
      <line
        :x1="center.x - axisLength"
        :y1="center.y"
        :x2="center.x + axisLength"
        :y2="center.y"
        stroke="#4b5563"
        stroke-width="1"
        stroke-dasharray="4"
      />
      <line
        :x1="center.x"
        :y1="center.y - axisLength"
        :x2="center.x"
        :y2="center.y + axisLength"
        stroke="#4b5563"
        stroke-width="1"
        stroke-dasharray="4"
      />

      <!-- Transformed axes -->
      <line
        :x1="center.x"
        :y1="center.y"
        :x2="transformPoint(gridRange, 0).x"
        :y2="transformPoint(gridRange, 0).y"
        stroke="#ef4444"
        stroke-width="1.5"
      />
      <line
        :x1="center.x"
        :y1="center.y"
        :x2="transformPoint(0, gridRange).x"
        :y2="transformPoint(0, gridRange).y"
        stroke="#22c55e"
        stroke-width="1.5"
      />

      <!-- Original vector -->
      <line
        :x1="center.x"
        :y1="center.y"
        :x2="center.x + vector.x * scale"
        :y2="center.y - vector.y * scale"
        stroke="#60a5fa"
        stroke-width="2"
      />
      <circle
        :cx="center.x + vector.x * scale"
        :cy="center.y - vector.y * scale"
        r="2"
        fill="#60a5fa"
      />

      <!-- Transformed vector -->
      <line
        :x1="center.x"
        :y1="center.y"
        :x2="transformPoint(vector.x, vector.y).x"
        :y2="transformPoint(vector.x, vector.y).y"
        stroke="#f59e0b"
        stroke-width="2"
      />
      <circle
        :cx="transformPoint(vector.x, vector.y).x"
        :cy="transformPoint(vector.x, vector.y).y"
        r="2"
        fill="#f59e0b"
      />

      <!-- Labels -->
      <text
        :x="center.x + vector.x * scale + 5"
        :y="center.y - vector.y * scale - 5"
        fill="#60a5fa"
        font-size="5"
      >
        v ({{ vector.x.toFixed(1) }}, {{ vector.y.toFixed(1) }})
      </text>
      <text
        :x="transformPoint(vector.x, vector.y).x + 5"
        :y="transformPoint(vector.x, vector.y).y - 5"
        fill="#f59e0b"
        font-size="5"
      >
        Av ({{ transformedVector.x.toFixed(1) }},
        {{ transformedVector.y.toFixed(1) }})
      </text>

      <!-- Axis labels -->
      <text
        :x="center.x + axisLength + 3"
        :y="center.y + 8"
        fill="#ef4444"
        font-size="5"
      >
        e₁
      </text>
      <text
        :x="center.x + 3"
        :y="center.y - axisLength - 3"
        fill="#22c55e"
        font-size="5"
      >
        e₂
      </text>
    </svg>

    <!-- Vector input -->
    <div class="flex gap-2 items-center">
      <div class="text-[10px] font-bold text-gray-300">Vector v:</div>
      <input
        type="number"
        :value="vector.x"
        @input="updateVector('x', $event)"
        class="w-10 px-0.5 py-0.5 bg-gray-700 border border-gray-600 rounded text-center text-white text-[10px]"
        step="0.1"
      />
      <input
        type="number"
        :value="vector.y"
        @input="updateVector('y', $event)"
        class="w-10 px-0.5 py-0.5 bg-gray-700 border border-gray-600 rounded text-center text-white text-[10px]"
        step="0.1"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from "vue";

const svgSize = 280;
const scale = 28;
const gridRange = 5;
const axisLength = 126;
const center = { x: svgSize / 2, y: svgSize / 2 };

const matrix = ref([
  [1, 0],
  [0, 1],
]);

const vector = ref({ x: 2, y: 1 });

const presets = [
  {
    name: "Identity",
    matrix: [
      [1, 0],
      [0, 1],
    ],
  },
  {
    name: "Scale 2x",
    matrix: [
      [2, 0],
      [0, 2],
    ],
  },
  {
    name: "Scale X",
    matrix: [
      [2, 0],
      [0, 1],
    ],
  },
  {
    name: "Rotate 45°",
    matrix: [
      [0.707, -0.707],
      [0.707, 0.707],
    ],
  },
  {
    name: "Rotate 90°",
    matrix: [
      [0, -1],
      [1, 0],
    ],
  },
  {
    name: "Shear",
    matrix: [
      [1, 1],
      [0, 1],
    ],
  },
  {
    name: "Reflect X",
    matrix: [
      [-1, 0],
      [0, 1],
    ],
  },
  {
    name: "Reflect Y",
    matrix: [
      [1, 0],
      [0, -1],
    ],
  },
];

const gridLines = computed(() => {
  const lines = [];
  for (let i = -gridRange; i <= gridRange; i++) {
    lines.push(i);
  }
  return lines;
});

const transformedVector = computed(() => {
  return {
    x:
      matrix.value[0][0] * vector.value.x + matrix.value[0][1] * vector.value.y,
    y:
      matrix.value[1][0] * vector.value.x + matrix.value[1][1] * vector.value.y,
  };
});

function transformPoint(x: number, y: number) {
  const tx = matrix.value[0][0] * x + matrix.value[0][1] * y;
  const ty = matrix.value[1][0] * x + matrix.value[1][1] * y;
  return {
    x: center.x + tx * scale,
    y: center.y - ty * scale,
  };
}

function updateMatrix(idx: number, event: Event) {
  const value = parseFloat((event.target as HTMLInputElement).value) || 0;
  const row = Math.floor(idx / 2);
  const col = idx % 2;
  matrix.value[row][col] = value;
}

function updateVector(coord: "x" | "y", event: Event) {
  const value = parseFloat((event.target as HTMLInputElement).value) || 0;
  vector.value[coord] = value;
}

function applyPreset(preset: { name: string; matrix: number[][] }) {
  matrix.value = preset.matrix.map((row) => [...row]);
}
</script>

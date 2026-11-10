/* Row-vector convention throughout. Hand-set weights, computed activations. */
(function (root) {
  'use strict';
  const words = ['the', 'cat', 'sat'];
  const E = [[1, 0, 1], [0, 1, 0], [1, 0, 0]];
  const P = [[0, 0, 0], [0, 0, 1], [0, 1, 0]];
  const WQ = [[0, 0], [1, 0], [0, 1]];
  const WK = [[-0.5, 1], [1.5, 0], [0.5, 0]];
  const WV = [[0.75, -0.25], [-0.25, 0.75], [0.25, 0.25]];
  const dot = (a, b) => a.reduce((sum, v, i) => sum + v * b[i], 0);
  const matmul = (a, b) => a.map(row => b[0].map((_, j) => dot(row, b.map(r => r[j]))));
  function compute({ mask = true, scale = true, equal = false, zeroValue = false } = {}) {
    const X = E.map((row, i) => row.map((v, j) => v + P[i][j]));
    const Q = matmul(X, WQ), K = matmul(X, WK), V = matmul(X, WV);
    if (zeroValue) V[1] = [0, 0];
    const scores = Q.map(q => K.map(k => dot(q, k)));
    const scaled = scores.map(row => row.map(v => v / (scale ? Math.sqrt(K[0].length) : 1)));
    const logits = scaled.map((row, i) => row.map((s, j) => mask && j > i ? -Infinity : equal ? 0 : s));
    const maxima = logits.map(row => Math.max(...row));
    const exponents = logits.map((row, i) => row.map(s => Math.exp(s - maxima[i])));
    const sums = exponents.map(row => row.reduce((a, b) => a + b, 0));
    const A = exponents.map((row, i) => row.map(e => e / sums[i]));
    const Z = matmul(A, V);
    return { X, Q, K, V, scores, scaled, logits, maxima, exponents, sums, A, Z };
  }
  const api = { words, E, P, WQ, WK, WV, dot, matmul, compute };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.AttentionExample = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);

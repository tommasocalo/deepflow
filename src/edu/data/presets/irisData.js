/**
 * Iris (Fisher, 1936): 150 flowers × 4 measurements in cm, plus the species index.
 *
 * Values are identical to scikit-learn's sklearn/datasets/data/iris.csv
 * (sha256 f13ffa8fdd56fd8e6c8d16d4081a3fbd3114bcd0aae4256c43205169cd9d1449), which load_iris() returns. This is NOT the UCI copy: UCI has two
 * wrong rows (1-based sample 35 should be 4.9, 3.1, 1.5, 0.2 and sample 38 should be 4.9, 3.6, 1.4, 0.1). The
 * tfjs-examples iris/data.js copies UCI and must not be used. scripts/edu-data-iris.test.mjs re-serialises these rows
 * and checks the hash.
 *
 * Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. Annals of Eugenics, 7(2), 179–188.
 * The 1936 measurements are treated as public domain. The file is distributed with scikit-learn, whose notice follows:
 *
 * BSD 3-Clause License
 *
 * Copyright (c) 2007-2026 The scikit-learn developers.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

export const IRIS_FEATURE_NAMES = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'];
export const IRIS_CLASS_NAMES = ['setosa', 'versicolor', 'virginica'];

/** One row per flower, in scikit-learn order: sepal length, sepal width, petal length, petal width, species. */
export const IRIS_ROWS = [
    [5.1, 3.5, 1.4, 0.2, 0],
    [4.9, 3.0, 1.4, 0.2, 0],
    [4.7, 3.2, 1.3, 0.2, 0],
    [4.6, 3.1, 1.5, 0.2, 0],
    [5.0, 3.6, 1.4, 0.2, 0],
    [5.4, 3.9, 1.7, 0.4, 0],
    [4.6, 3.4, 1.4, 0.3, 0],
    [5.0, 3.4, 1.5, 0.2, 0],
    [4.4, 2.9, 1.4, 0.2, 0],
    [4.9, 3.1, 1.5, 0.1, 0],
    [5.4, 3.7, 1.5, 0.2, 0],
    [4.8, 3.4, 1.6, 0.2, 0],
    [4.8, 3.0, 1.4, 0.1, 0],
    [4.3, 3.0, 1.1, 0.1, 0],
    [5.8, 4.0, 1.2, 0.2, 0],
    [5.7, 4.4, 1.5, 0.4, 0],
    [5.4, 3.9, 1.3, 0.4, 0],
    [5.1, 3.5, 1.4, 0.3, 0],
    [5.7, 3.8, 1.7, 0.3, 0],
    [5.1, 3.8, 1.5, 0.3, 0],
    [5.4, 3.4, 1.7, 0.2, 0],
    [5.1, 3.7, 1.5, 0.4, 0],
    [4.6, 3.6, 1.0, 0.2, 0],
    [5.1, 3.3, 1.7, 0.5, 0],
    [4.8, 3.4, 1.9, 0.2, 0],
    [5.0, 3.0, 1.6, 0.2, 0],
    [5.0, 3.4, 1.6, 0.4, 0],
    [5.2, 3.5, 1.5, 0.2, 0],
    [5.2, 3.4, 1.4, 0.2, 0],
    [4.7, 3.2, 1.6, 0.2, 0],
    [4.8, 3.1, 1.6, 0.2, 0],
    [5.4, 3.4, 1.5, 0.4, 0],
    [5.2, 4.1, 1.5, 0.1, 0],
    [5.5, 4.2, 1.4, 0.2, 0],
    [4.9, 3.1, 1.5, 0.2, 0],
    [5.0, 3.2, 1.2, 0.2, 0],
    [5.5, 3.5, 1.3, 0.2, 0],
    [4.9, 3.6, 1.4, 0.1, 0],
    [4.4, 3.0, 1.3, 0.2, 0],
    [5.1, 3.4, 1.5, 0.2, 0],
    [5.0, 3.5, 1.3, 0.3, 0],
    [4.5, 2.3, 1.3, 0.3, 0],
    [4.4, 3.2, 1.3, 0.2, 0],
    [5.0, 3.5, 1.6, 0.6, 0],
    [5.1, 3.8, 1.9, 0.4, 0],
    [4.8, 3.0, 1.4, 0.3, 0],
    [5.1, 3.8, 1.6, 0.2, 0],
    [4.6, 3.2, 1.4, 0.2, 0],
    [5.3, 3.7, 1.5, 0.2, 0],
    [5.0, 3.3, 1.4, 0.2, 0],
    [7.0, 3.2, 4.7, 1.4, 1],
    [6.4, 3.2, 4.5, 1.5, 1],
    [6.9, 3.1, 4.9, 1.5, 1],
    [5.5, 2.3, 4.0, 1.3, 1],
    [6.5, 2.8, 4.6, 1.5, 1],
    [5.7, 2.8, 4.5, 1.3, 1],
    [6.3, 3.3, 4.7, 1.6, 1],
    [4.9, 2.4, 3.3, 1.0, 1],
    [6.6, 2.9, 4.6, 1.3, 1],
    [5.2, 2.7, 3.9, 1.4, 1],
    [5.0, 2.0, 3.5, 1.0, 1],
    [5.9, 3.0, 4.2, 1.5, 1],
    [6.0, 2.2, 4.0, 1.0, 1],
    [6.1, 2.9, 4.7, 1.4, 1],
    [5.6, 2.9, 3.6, 1.3, 1],
    [6.7, 3.1, 4.4, 1.4, 1],
    [5.6, 3.0, 4.5, 1.5, 1],
    [5.8, 2.7, 4.1, 1.0, 1],
    [6.2, 2.2, 4.5, 1.5, 1],
    [5.6, 2.5, 3.9, 1.1, 1],
    [5.9, 3.2, 4.8, 1.8, 1],
    [6.1, 2.8, 4.0, 1.3, 1],
    [6.3, 2.5, 4.9, 1.5, 1],
    [6.1, 2.8, 4.7, 1.2, 1],
    [6.4, 2.9, 4.3, 1.3, 1],
    [6.6, 3.0, 4.4, 1.4, 1],
    [6.8, 2.8, 4.8, 1.4, 1],
    [6.7, 3.0, 5.0, 1.7, 1],
    [6.0, 2.9, 4.5, 1.5, 1],
    [5.7, 2.6, 3.5, 1.0, 1],
    [5.5, 2.4, 3.8, 1.1, 1],
    [5.5, 2.4, 3.7, 1.0, 1],
    [5.8, 2.7, 3.9, 1.2, 1],
    [6.0, 2.7, 5.1, 1.6, 1],
    [5.4, 3.0, 4.5, 1.5, 1],
    [6.0, 3.4, 4.5, 1.6, 1],
    [6.7, 3.1, 4.7, 1.5, 1],
    [6.3, 2.3, 4.4, 1.3, 1],
    [5.6, 3.0, 4.1, 1.3, 1],
    [5.5, 2.5, 4.0, 1.3, 1],
    [5.5, 2.6, 4.4, 1.2, 1],
    [6.1, 3.0, 4.6, 1.4, 1],
    [5.8, 2.6, 4.0, 1.2, 1],
    [5.0, 2.3, 3.3, 1.0, 1],
    [5.6, 2.7, 4.2, 1.3, 1],
    [5.7, 3.0, 4.2, 1.2, 1],
    [5.7, 2.9, 4.2, 1.3, 1],
    [6.2, 2.9, 4.3, 1.3, 1],
    [5.1, 2.5, 3.0, 1.1, 1],
    [5.7, 2.8, 4.1, 1.3, 1],
    [6.3, 3.3, 6.0, 2.5, 2],
    [5.8, 2.7, 5.1, 1.9, 2],
    [7.1, 3.0, 5.9, 2.1, 2],
    [6.3, 2.9, 5.6, 1.8, 2],
    [6.5, 3.0, 5.8, 2.2, 2],
    [7.6, 3.0, 6.6, 2.1, 2],
    [4.9, 2.5, 4.5, 1.7, 2],
    [7.3, 2.9, 6.3, 1.8, 2],
    [6.7, 2.5, 5.8, 1.8, 2],
    [7.2, 3.6, 6.1, 2.5, 2],
    [6.5, 3.2, 5.1, 2.0, 2],
    [6.4, 2.7, 5.3, 1.9, 2],
    [6.8, 3.0, 5.5, 2.1, 2],
    [5.7, 2.5, 5.0, 2.0, 2],
    [5.8, 2.8, 5.1, 2.4, 2],
    [6.4, 3.2, 5.3, 2.3, 2],
    [6.5, 3.0, 5.5, 1.8, 2],
    [7.7, 3.8, 6.7, 2.2, 2],
    [7.7, 2.6, 6.9, 2.3, 2],
    [6.0, 2.2, 5.0, 1.5, 2],
    [6.9, 3.2, 5.7, 2.3, 2],
    [5.6, 2.8, 4.9, 2.0, 2],
    [7.7, 2.8, 6.7, 2.0, 2],
    [6.3, 2.7, 4.9, 1.8, 2],
    [6.7, 3.3, 5.7, 2.1, 2],
    [7.2, 3.2, 6.0, 1.8, 2],
    [6.2, 2.8, 4.8, 1.8, 2],
    [6.1, 3.0, 4.9, 1.8, 2],
    [6.4, 2.8, 5.6, 2.1, 2],
    [7.2, 3.0, 5.8, 1.6, 2],
    [7.4, 2.8, 6.1, 1.9, 2],
    [7.9, 3.8, 6.4, 2.0, 2],
    [6.4, 2.8, 5.6, 2.2, 2],
    [6.3, 2.8, 5.1, 1.5, 2],
    [6.1, 2.6, 5.6, 1.4, 2],
    [7.7, 3.0, 6.1, 2.3, 2],
    [6.3, 3.4, 5.6, 2.4, 2],
    [6.4, 3.1, 5.5, 1.8, 2],
    [6.0, 3.0, 4.8, 1.8, 2],
    [6.9, 3.1, 5.4, 2.1, 2],
    [6.7, 3.1, 5.6, 2.4, 2],
    [6.9, 3.1, 5.1, 2.3, 2],
    [5.8, 2.7, 5.1, 1.9, 2],
    [6.8, 3.2, 5.9, 2.3, 2],
    [6.7, 3.3, 5.7, 2.5, 2],
    [6.7, 3.0, 5.2, 2.3, 2],
    [6.3, 2.5, 5.0, 1.9, 2],
    [6.5, 3.0, 5.2, 2.0, 2],
    [6.2, 3.4, 5.4, 2.3, 2],
    [5.9, 3.0, 5.1, 1.8, 2],
];

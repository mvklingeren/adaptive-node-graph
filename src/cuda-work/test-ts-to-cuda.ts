import { tsToCuda } from "./ts-to-cuda";

type TestResult = {
  success: boolean;
  name: string;
  message?: string;
};

function runTest(
  name: string,
  tsCode: string,
  expectedCudaCode: string
): TestResult {
  try {
    const { deviceCode } = tsToCuda(tsCode);
    const cleanedDeviceCode = deviceCode.trim().replace(/\s+/g, " ");
    const cleanedExpectedCudaCode = expectedCudaCode.trim().replace(/\s+/g, " ");

    if (cleanedDeviceCode === cleanedExpectedCudaCode) {
      return { success: true, name };
    } else {
      return {
        success: false,
        name,
        message: `\nExpected:\n${expectedCudaCode}\nGot:\n${deviceCode}`,
      };
    }
  } catch (e: any) {
    return { success: false, name, message: e.message };
  }
}

function runErrorTest(
  name: string,
  tsCode: string,
  expectedErrorMessage: string
): TestResult {
  try {
    tsToCuda(tsCode);
    return {
      success: false,
      name,
      message: "Expected an error, but none was thrown.",
    };
  } catch (e: any) {
    if (e.message.includes(expectedErrorMessage)) {
      return { success: true, name };
    } else {
      return {
        success: false,
        name,
        message: `\nExpected error message:\n${expectedErrorMessage}\nGot:\n${e.message}`,
      };
    }
  }
}

const tests = [
  runTest(
    "Simple function with one input",
    `function add(a: number, out: number) { out(0) = a(0) + 1; }`,
    `__device__ void add(Tensor<float> out, const Tensor<float> a) {
  out(0) = a(0) + 1;
}`
  ),
  runTest(
    "Global function with @cuda global",
    `/** @cuda global */
function myKernel(a: number, out: number) { out(0) = a(0) * 2; }`,
    `__global__ void myKernel(Tensor<float> out, const Tensor<float> a) {
  out(0) = a(0) * 2;
}`
  ),
  runTest(
    "Variable declaration with type",
    `function test(out: number) { let x: number = 1.0; out(0) = x; }`,
    `__device__ void test(Tensor<float> out) {
  float x = 1.0;
  out(0) = x;
}`
  ),
  runTest(
    "Variable declaration with inferred int type",
    `function test(out: number) { let x = 1; out(0) = x; }`,
    `__device__ void test(Tensor<float> out) {
  int x = 1;
  out(0) = x;
}`
  ),
  runTest(
    "If statement",
    `function test(a: number, out: number) { if (a(0) > 0) { out(0) = 1; } }`,
    `__device__ void test(Tensor<float> out, const Tensor<float> a) {
  if (a(0) > 0) {
    out(0) = 1;
  }
}`
  ),
  runTest(
    "If-else statement",
    `function test(a: number, out: number) { if (a(0) > 0) { out(0) = 1; } else { out(0) = 0; } }`,
    `__device__ void test(Tensor<float> out, const Tensor<float> a) {
  if (a(0) > 0) {
    out(0) = 1;
  } else {
    out(0) = 0;
  }
}`
  ),
  runTest(
    "If-else if-else statement",
    `function test(a: number, out: number) { if (a(0) > 1) { out(0) = 1; } else if (a(0) < 0) { out(0) = -1; } else { out(0) = 0; } }`,
    `__device__ void test(Tensor<float> out, const Tensor<float> a) {
  if (a(0) > 1) {
    out(0) = 1;
  } else if (a(0) < 0) {
    out(0) = -1;
  } else {
    out(0) = 0;
  }
}`
  ),
  runTest(
    "For loop",
    `function test(out: number) { for (let i: number = 0; i < 10; i++) { out(i) = i; } }`,
    `__device__ void test(Tensor<float> out) {
  for (float i = 0; i < 10; i++) {
    out(i) = i;
  }
}`
  ),
  runTest(
    "While loop",
    `function test(out: number) { let i = 0; while (i < 10) { out(i) = i; i++; } }`,
    `__device__ void test(Tensor<float> out) {
  int i = 0;
  while (i < 10) {
    out(i) = i;
    i++;
  }
}`
  ),
  runTest(
    "Boolean type",
    `function test(a: boolean, out: boolean) { out(0) = !a(0); }`,
    `__device__ void test(Tensor<bool> out, const Tensor<bool> a) {
  out(0) = !a(0);
}`
  ),
  runErrorTest(
    "Function with no parameters",
    `function noParams() {}`,
    "Function must have at least one parameter (output)."
  ),
];

let failures = 0;
for (const result of tests) {
  if (result.success) {
    console.log(`✅ ${result.name}`);
  } else {
    failures++;
    console.error(`❌ ${result.name}`);
    if (result.message) {
      console.error(result.message);
    }
  }
}

if (failures > 0) {
  console.error(`\n${failures} test(s) failed.`);
  process.exit(1);
} else {
  console.log("\nAll tests passed!");
}

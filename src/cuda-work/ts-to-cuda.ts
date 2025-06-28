import * as ts from "typescript";

export function tsToCuda(code: string): {
  functionName: string;
  deviceCode: string;
} {
  const sourceFile = ts.createSourceFile(
    "temp.ts",
    code,
    ts.ScriptTarget.Latest,
    true
  );

  let functionName = "";
  let deviceCode = "";

  function visit(node: ts.Node) {
    if (ts.isFunctionDeclaration(node) && node.name) {
      functionName = node.name.text;
      if (node.parameters.length === 0) return;

      const inputParams = node.parameters.slice(0, -1);
      const outputParam = node.parameters[node.parameters.length - 1];

      const mapParam = (p: ts.ParameterDeclaration, isOutput: boolean) => {
        const type = p.type ? tsToCudaType(p.type) : "float";
        const name = p.name.getText(sourceFile);
        return isOutput
          ? `Tensor<${type}> ${name}`
          : `const Tensor<${type}> ${name}`;
      };

      const paramStrings = [
        mapParam(outputParam, true),
        ...inputParams.map((p) => mapParam(p, false)),
      ];

      const body = processBlock(node.body!);
      deviceCode = `__device__ void ${functionName}(${paramStrings.join(", ")}) {\n${body}\n}`;
    }
    ts.forEachChild(node, visit);
  }

  function processBlock(block: ts.Block): string {
    let bodyCode = "";
    block.statements.forEach((stmt) => {
      bodyCode += processStatement(stmt);
    });
    return bodyCode;
  }

  function processStatement(stmt: ts.Statement): string {
    if (ts.isExpressionStatement(stmt)) {
      return "  " + processExpression(stmt.expression) + ";\n";
    }
    if (ts.isVariableStatement(stmt)) {
      let result = "  ";
      for (const decl of stmt.declarationList.declarations) {
        const name = decl.name.getText(sourceFile);
        const type = "int"; // Simplified for now
        result += `${type} ${name}`;
        if (decl.initializer) {
          result += ` = ${processExpression(decl.initializer)}`;
        }
        result += ";\n";
      }
      return result;
    }
    return `// Unsupported statement: ${ts.SyntaxKind[stmt.kind]}\n`;
  }

  function processExpression(expr: ts.Expression): string {
    if (ts.isBinaryExpression(expr)) {
      const left = processExpression(expr.left);
      const right = processExpression(expr.right);
      const operator = expr.operatorToken.getText(sourceFile);
      return `${left} ${operator} ${right}`;
    }
    if (ts.isCallExpression(expr)) {
      const callee = processExpression(expr.expression);
      const args = expr.arguments.map(processExpression).join(", ");
      return `${callee}(${args})`;
    }
    if (ts.isElementAccessExpression(expr)) {
      const element = processExpression(expr.expression);
      const index = processExpression(expr.argumentExpression);
      return `${element}(${index})`;
    }
    if (ts.isIdentifier(expr)) {
      return expr.text;
    }
    if (ts.isNumericLiteral(expr)) {
      return expr.text;
    }
    return `/* Unsupported expression: ${ts.SyntaxKind[expr.kind]} */`;
  }

  function tsToCudaType(typeNode: ts.TypeNode): string {
    switch (typeNode.kind) {
      case ts.SyntaxKind.NumberKeyword:
        return "float";
      case ts.SyntaxKind.StringKeyword:
        return "char*";
      default:
        return "void";
    }
  }

  visit(sourceFile);
  return { functionName, deviceCode };
}

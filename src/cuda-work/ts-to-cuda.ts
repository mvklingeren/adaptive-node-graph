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

  function visit(node: ts.Node): { functionName: string; deviceCode: string } | null {
    if (ts.isFunctionDeclaration(node) && node.name) {
      const functionName = node.name.text;
      if (node.parameters.length === 0)
        throw new Error("Function must have at least one parameter (output).");

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

      if (!node.body) {
        throw new Error(`Function ${functionName} must have a body.`);
      }

      const body = processBlock(node.body);
      const leadingComments = ts.getLeadingCommentRanges(code, node.pos);
      const isGlobal = leadingComments?.some(comment => code.substring(comment.pos, comment.end).includes("@cuda global"));

      const kernelType = isGlobal ? "__global__" : "__device__";
      const deviceCode = `${kernelType} void ${functionName}(${paramStrings.join(
        ", "
      )}) {\n${body}\n}`;
      return { functionName, deviceCode };
    }

    for (const child of node.getChildren(sourceFile)) {
      const result = visit(child);
      if (result) return result;
    }
    return null;
  }

  function processBlock(block: ts.Block): string {
    return block.statements.map(processStatement).join("");
  }

  function processStatement(stmt: ts.Statement): string {
    if (ts.isExpressionStatement(stmt)) {
      return "  " + processExpression(stmt.expression) + ";\n";
    }
    if (ts.isVariableStatement(stmt)) {
      let result = "  ";
      for (const decl of stmt.declarationList.declarations) {
        const name = decl.name.getText(sourceFile);
        const type = decl.type
          ? tsToCudaType(decl.type)
          : inferTypeFromExpression(decl.initializer);
        result += `${type} ${name}`;
        if (decl.initializer) {
          result += ` = ${processExpression(decl.initializer)}`;
        }
        result += ";\n";
      }
      return result;
    }
    if (ts.isIfStatement(stmt)) {
      const condition = processExpression(stmt.expression);
      const thenStatement = processBlock(stmt.thenStatement as ts.Block);
      let result = `  if (${condition}) {\n${thenStatement}  }\n`;
      if (stmt.elseStatement) {
        if (ts.isIfStatement(stmt.elseStatement)) {
          // Handle 'else if'
          const elseIfResult = processStatement(stmt.elseStatement).trimStart();
          result = result.trimEnd() + ` else ${elseIfResult}\n`;
        } else {
          const elseStatement = processBlock(stmt.elseStatement as ts.Block);
          result += `  else {\n${elseStatement}  }\n`;
        }
      }
      return result;
    }
    if (ts.isForStatement(stmt)) {
      const initializer = stmt.initializer
        ? processVariableDeclarationList(stmt.initializer as ts.VariableDeclarationList)
        : "";
      const condition = stmt.condition ? processExpression(stmt.condition) : "";
      const incrementor = stmt.incrementor ? processExpression(stmt.incrementor) : "";
      const body = processBlock(stmt.statement as ts.Block);
      return `  for (${initializer}; ${condition}; ${incrementor}) {\n${body}  }\n`;
    }
    if (ts.isWhileStatement(stmt)) {
      const condition = processExpression(stmt.expression);
      const body = processBlock(stmt.statement as ts.Block);
      return `  while (${condition}) {\n${body}  }\n`;
    }
    throw new Error(`Unsupported statement: ${ts.SyntaxKind[stmt.kind]}`);
  }

  function processVariableDeclarationList(list: ts.VariableDeclarationList): string {
    return list.declarations
      .map((decl) => {
        const name = decl.name.getText(sourceFile);
        const type = decl.type
          ? tsToCudaType(decl.type)
          : inferTypeFromExpression(decl.initializer);
        let result = `${type} ${name}`;
        if (decl.initializer) {
          result += ` = ${processExpression(decl.initializer)}`;
        }
        return result;
      })
      .join(", ");
  }

  function inferTypeFromExpression(expr?: ts.Expression): string {
    if (!expr) return "float"; // Default type if no initializer

    if (ts.isNumericLiteral(expr)) {
      const text = expr.getText(sourceFile);
      console.log(`Inferring type for numeric literal: ${text}`);
      return text.includes(".") ? "float" : "int";
    }
    if (ts.isStringLiteral(expr)) {
      return "char*";
    }
    if (expr.kind === ts.SyntaxKind.TrueKeyword || expr.kind === ts.SyntaxKind.FalseKeyword) {
      return "bool";
    }
    // Add more complex inference rules as needed
    return "float"; // Default fallback
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
      return expr.getText(sourceFile);
    }
    if (ts.isPrefixUnaryExpression(expr)) {
      const operand = processExpression(expr.operand);
      const operator = ts.tokenToString(expr.operator);
      return `${operator}${operand}`;
    }
    if (ts.isPostfixUnaryExpression(expr)) {
      const operand = processExpression(expr.operand);
      const operator = ts.tokenToString(expr.operator);
      return `${operand}${operator}`;
    }
    if (expr.kind === ts.SyntaxKind.TrueKeyword) {
      return "true";
    }
    if (expr.kind === ts.SyntaxKind.FalseKeyword) {
      return "false";
    }
    throw new Error(`Unsupported expression: ${ts.SyntaxKind[expr.kind]}`);
  }

  function tsToCudaType(typeNode: ts.TypeNode): string {
    switch (typeNode.kind) {
      case ts.SyntaxKind.NumberKeyword:
        return "float";
      case ts.SyntaxKind.BooleanKeyword:
        return "bool";
      case ts.SyntaxKind.StringKeyword:
        return "char*";
      default:
        return "void";
    }
  }

  const result = visit(sourceFile);
  if (!result) {
    throw new Error("No function declaration found in the provided code.");
  }
  return result;
}

package com.example;

import com.github.javaparser.ParserConfiguration;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.ImportDeclaration;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.body.*;
import com.github.javaparser.ast.expr.*;
import com.github.javaparser.ast.stmt.*;
import com.github.javaparser.resolution.declarations.*;
import com.github.javaparser.resolution.types.ResolvedPrimitiveType;
import com.github.javaparser.resolution.types.ResolvedReferenceType;
import com.github.javaparser.resolution.types.ResolvedType;
import com.github.javaparser.resolution.MethodUsage;
import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ClassLoaderTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JarTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JavaParserTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.Objects;

/**
 * Usage:
 *   java -cp target/classes:deps.jar com.example.GroundedSymbolTable \
 *        --file   /path/to/checkout/src/Foo.java \
 *        --line   145 \
 *        --source-root /path/to/checkout/source \
 *        --classpath "$(defects4j export -p cp.compile -w /path/to/checkout)" \
 *        --out symbols/Chart_12.json --pretty
 * </pre>
 *
 * {@code --line} is 1-based and refers to the file as it exists in the
 * checkout. The Python driver already locates the buggy function by text, so
 * it can convert that offset to a line number rather than trusting the
 * dataset's line numbers, which do not match the checkout.
 */
public class GroundedSymbolTable {

    // ------------------------------------------------------------------
    // configuration
    // ------------------------------------------------------------------

    public static class Config {
        String file;
        int line = -1;
        List<String> sourceRoots = new ArrayList<>();
        List<String> classpath = new ArrayList<>();
        String out = null;
        boolean pretty = false;
        int maxMembersPerType = 4000;
        String languageLevel = "JAVA_11";
    }

    /** Counters so the caller can tell a rich table from an empty one. */
    static class Health {
        int receiversAttempted = 0;
        int receiversResolved = 0;
        int typesFailed = 0;
        final List<String> errors = new ArrayList<>();

        void error(String where, Throwable t) {
            typesFailed++;
            if (errors.size() < 40) {
                errors.add(where + ": " + t.getClass().getSimpleName()
                        + (t.getMessage() == null ? "" : " " + trim(t.getMessage(), 160)));
            }
        }
    }

    static String trim(String s, int n) {
        if (s == null) return "";
        s = s.replace('\n', ' ');
        return s.length() <= n ? s : s.substring(0, n) + "...";
    }

    // ------------------------------------------------------------------
    // entry point
    // ------------------------------------------------------------------

    public static void main(String[] args) throws Exception {
        Config cfg = parseArgs(args);
        if (cfg == null) {
            usage();
            System.exit(2);
            return;
        }
        JSONObject result = new GroundedSymbolTable().build(cfg);
        String text = cfg.pretty ? result.toString(2) : result.toString();
        if (cfg.out != null) {
            Path p = Paths.get(cfg.out);
            if (p.getParent() != null) Files.createDirectories(p.getParent());
            Files.write(p, text.getBytes(StandardCharsets.UTF_8));
            System.err.println("wrote " + p);
        } else {
            System.out.println(text);
        }
    }

    static void usage() {
        System.err.println(
            "usage: GroundedSymbolTable --file <Foo.java> --line <n>\n" +
            "         [--source-root <dir>]...   source roots for project types\n" +
            "         [--classpath <a:b:c>]      jars/class dirs (defects4j export -p cp.compile)\n" +
            "         [--out <file.json>] [--pretty]\n" +
            "         [--language-level JAVA_8|JAVA_11|JAVA_17|RAW]\n" +
            "         [--max-members N]");
    }

    static Config parseArgs(String[] a) {
        Config c = new Config();
        for (int i = 0; i < a.length; i++) {
            switch (a[i]) {
                case "--file":            c.file = a[++i]; break;
                case "--line":            c.line = Integer.parseInt(a[++i]); break;
                case "--source-root":     c.sourceRoots.add(a[++i]); break;
                case "--classpath":       c.classpath.addAll(splitClasspath(a[++i])); break;
                case "--out":             c.out = a[++i]; break;
                case "--pretty":          c.pretty = true; break;
                case "--max-members":     c.maxMembersPerType = Integer.parseInt(a[++i]); break;
                case "--language-level":  c.languageLevel = a[++i]; break;
                default:
                    System.err.println("unknown option: " + a[i]);
                    return null;
            }
        }
        if (c.file == null || c.line < 1) return null;
        return c;
    }

    public static List<String> splitClasspath(String cp) {
        List<String> out = new ArrayList<>();
        for (String part : cp.split(File.pathSeparator.equals(":") ? "[:;]" : ";")) {
            if (!part.trim().isEmpty()) out.add(part.trim());
        }
        return out;
    }

    // ------------------------------------------------------------------
    // main build
    // ------------------------------------------------------------------

    private final Health health = new Health();

    public JSONObject build(Config cfg) throws Exception {
        JSONObject root = new JSONObject();
        root.put("file", cfg.file);
        root.put("line", cfg.line);

        CombinedTypeSolver solver = buildTypeSolver(cfg, root);

        ParserConfiguration pc = new ParserConfiguration()
                .setSymbolResolver(new JavaSymbolSolver(solver));
        try {
            pc.setLanguageLevel(ParserConfiguration.LanguageLevel.valueOf(cfg.languageLevel));
        } catch (IllegalArgumentException ex) {
            pc.setLanguageLevel(ParserConfiguration.LanguageLevel.JAVA_11);
        }
        StaticJavaParser.setConfiguration(pc);

        CompilationUnit cu = StaticJavaParser.parse(new File(cfg.file));
        cu.getPackageDeclaration().ifPresent(pd -> root.put("package", pd.getNameAsString()));

        Node hole = deepestNodeAtLine(cu, cfg.line);
        if (hole == null) {
            root.put("error", "no AST node covers line " + cfg.line);
            root.put("health", healthJson());
            return root;
        }

        CallableDeclaration<?> method = enclosing(hole, CallableDeclaration.class);
        TypeDeclaration<?> type = enclosing(hole, TypeDeclaration.class);
        if (method != null) root.put("method", method.getNameAsString());
        if (type != null) root.put("class", type.getNameAsString());

        // ---- names visible in a bare identifier position ---------------
        Set<String> bare = new TreeSet<>();
        JSONObject scope = new JSONObject();

        Map<String, ResolvedType> inScopeVars = new LinkedHashMap<>();
        // Set<String> usedTypeNames = new LinkedHashSet<>();
        Map<String, String> importedTypes = new LinkedHashMap<>();
        collectLocalsAndParams(hole, cfg.line, inScopeVars, bare, scope);

        JSONObject receivers = new JSONObject();
        collectThisAndFields(type, receivers, inScopeVars, bare, scope, cfg);
        collectImportsAndStatics(cu, bare, scope, importedTypes);

                // ---- static members of type names: Branch.ON_EX, Token.INC, Math.max ----
        Set<String> hiding = new HashSet<>(inScopeVars.keySet());   // a variable or field
        JSONArray fieldNames = scope.optJSONArray("fields");          // with the same name
        if (fieldNames != null) {                                      // hides the type
            for (int i = 0; i < fieldNames.length(); i++) hiding.add(fieldNames.optString(i));
        }
        JSONObject typeMembers = new JSONObject();
        for (Map.Entry<String, String> e : typeNameCandidates(cu, importedTypes).entrySet()) {
            if (hiding.contains(e.getKey())) continue;
            JSONObject members = staticMembersForTypeName(e.getValue(), solver, cfg);
            if (members != null) typeMembers.put(e.getKey(), members);
        }
        scope.put("type_members", typeMembers);


        scope.put("type_members", typeMembers);

        // ---- members reachable through each in-scope receiver ----------
        for (Map.Entry<String, ResolvedType> e : inScopeVars.entrySet()) {
            health.receiversAttempted++;
            JSONObject members = membersOf(e.getValue(), cfg);
            if (members != null) {
                health.receiversResolved++;
                receivers.put(e.getKey(), members);
            }
        }

        // ---- receivers that appear as expressions in the method --------
        // if (method != null) collectExpressionReceivers(method, receivers, cfg);

        // ---- enum constants of any switch subject ----------------------
        JSONObject enums = new JSONObject();
        if (method != null) collectSwitchEnums(method, enums, bare);
        collectHoleSwitch(hole, root);
        for (Map.Entry<String, ResolvedType> e : inScopeVars.entrySet()) {
            addEnumConstants(e.getValue(), enums, bare);
        }

        // ---- the union -------------------------------------------
        Set<String> all = new TreeSet<>(bare);
        for (String k : receivers.keySet()) {
            JSONObject r = receivers.getJSONObject(k);
            for (String field : new String[]{"methods", "fields"}) {
                JSONArray arr = r.optJSONArray(field);
                if (arr == null) continue;
                for (int i = 0; i < arr.length(); i++) {
                    Object o = arr.get(i);
                    all.add(o instanceof JSONObject ? ((JSONObject) o).getString("name")
                                                    : String.valueOf(o));
                }
            }
        }
        for (String k : enums.keySet()) {
            JSONArray arr = enums.getJSONArray(k);
            for (int i = 0; i < arr.length(); i++) all.add(arr.getString(i));
        }

        root.put("scope", scope);
        root.put("after_dot", receivers);
        root.put("enums", enums);
        root.put("bare", new JSONArray(bare));
        root.put("all_identifiers", new JSONArray(all));
        root.put("health", healthJson());
        return root;
    }

    private JSONObject healthJson() {
        JSONObject h = new JSONObject();
        h.put("receivers_attempted", health.receiversAttempted);
        h.put("receivers_resolved", health.receiversResolved);
        h.put("resolution_rate", health.receiversAttempted == 0 ? 0.0
                : (double) health.receiversResolved / health.receiversAttempted);
        h.put("failures", health.typesFailed);
        h.put("errors", new JSONArray(health.errors));
        return h;
    }

    // ------------------------------------------------------------------
    // type solver
    // ------------------------------------------------------------------

    private CombinedTypeSolver buildTypeSolver(Config cfg, JSONObject root) {
        CombinedTypeSolver ts = new CombinedTypeSolver();
        JSONArray used = new JSONArray();

        // Project sources first: a class present in the checkout should resolve
        // to the checked-out version, not to some jar of a different release.
        for (String r : cfg.sourceRoots) {
            File f = new File(r);
            if (f.isDirectory()) {
                ts.add(new JavaParserTypeSolver(f));
                used.put("src:" + r);
            }
        }

        List<URL> dirUrls = new ArrayList<>();
        for (String cp : cfg.classpath) {
            File f = new File(cp);
            if (!f.exists()) continue;
            try {
                if (f.isFile() && cp.endsWith(".jar")) {
                    ts.add(new JarTypeSolver(f));
                    used.put("jar:" + cp);
                } else if (f.isDirectory()) {
                    dirUrls.add(f.toURI().toURL());
                    used.put("dir:" + cp);
                }
            } catch (Exception ex) {
                health.error("classpath " + cp, ex);
            }
        }
        if (!dirUrls.isEmpty()) {
            // Directories of .class files have no dedicated solver; a
            // classloader over them serves the same purpose.
            URLClassLoader cl = new URLClassLoader(dirUrls.toArray(new URL[0]),
                    GroundedSymbolTable.class.getClassLoader());
            ts.add(new ClassLoaderTypeSolver(cl));
        }

        ts.add(new ReflectionTypeSolver(false));   // JDK, last
        root.put("type_solvers", used);
        return ts;
    }

    // ------------------------------------------------------------------
    // locating the hole
    // ------------------------------------------------------------------

    /** Find the smallest AST node covering the requested source line. (DFS-like traversal) */
    static Node deepestNodeAtLine(Node root, int line) {
        Node best = null;
        Deque<Node> stack = new ArrayDeque<>();
        stack.push(root);
        while (!stack.isEmpty()) {
            Node n = stack.pop();
            if (n.getRange().isPresent()) {
                int b = n.getRange().get().begin.line, e = n.getRange().get().end.line;
                if (b <= line && line <= e) {
                    if (best == null || span(n) <= span(best)) best = n;
                }
            }
            n.getChildNodes().forEach(stack::push);
        }
        return best;
    }

    static int span(Node n) {
        return n.getRange().map(r -> r.end.line - r.begin.line).orElse(Integer.MAX_VALUE);
    }

    @SuppressWarnings("unchecked")
    static <T extends Node> T enclosing(Node n, Class<T> cls) {
        Node cur = n;
        while (cur != null) {
            if (cls.isInstance(cur)) return (T) cur;
            cur = cur.getParentNode().orElse(null);
        }
        return null;
    }

    // ------------------------------------------------------------------
    // scope collection
    // ------------------------------------------------------------------

    /**
     * Locals and parameters visible at the hole. A local counts only if its
     * declaration ENDS before the hole line -- a variable declared after the
     * hole is not in scope there, and offering it to the decoder would invite
     * "cannot find symbol" from the other direction.
     */
    private void collectLocalsAndParams(Node hole, int line,
                                        Map<String, ResolvedType> vars,
                                        Set<String> bare, JSONObject scope) {
        JSONArray locals = new JSONArray();
        JSONArray params = new JSONArray();

        Node cur = hole;
        while (cur != null) {
            if (cur instanceof BlockStmt) {
                for (Statement st : ((BlockStmt) cur).getStatements()) {
                    if (st.getRange().isPresent() && st.getRange().get().end.line >= line) continue;
                    st.findAll(VariableDeclarator.class)
                      .forEach(v -> addVar(v.getNameAsString(), v, vars, bare, locals));
                }
            } else if (cur instanceof ForStmt) {
                ((ForStmt) cur).getInitialization().forEach(e ->
                    e.findAll(VariableDeclarator.class)
                     .forEach(v -> addVar(v.getNameAsString(), v, vars, bare, locals)));
            } else if (cur instanceof ForEachStmt) {
                VariableDeclarator v = ((ForEachStmt) cur).getVariable().getVariable(0);
                addVar(v.getNameAsString(), v, vars, bare, locals);
            } else if (cur instanceof CatchClause) {
                Parameter p = ((CatchClause) cur).getParameter();
                addParam(p, vars, bare, params);
            } else if (cur instanceof LambdaExpr) {
                ((LambdaExpr) cur).getParameters().forEach(p -> addParam(p, vars, bare, params));
            } else if (cur instanceof CallableDeclaration) {
                ((CallableDeclaration<?>) cur).getParameters()
                        .forEach(p -> addParam(p, vars, bare, params));
            }
            cur = cur.getParentNode().orElse(null);
        }
        scope.put("locals", locals);
        scope.put("params", params);
    }

    private void addVar(String name, VariableDeclarator v,
                        Map<String, ResolvedType> vars, Set<String> bare, JSONArray out) {
        bare.add(name);
        JSONObject o = new JSONObject().put("name", name);
        try {
            ResolvedType t = v.getType().resolve();
            vars.put(name, t);
            o.put("type", describe(t));
        } catch (Throwable ex) {
            o.put("type", v.getTypeAsString());
            o.put("unresolved", true);
            health.error("local " + name, ex);
        }
        out.put(o);
    }

    private void addParam(Parameter p, Map<String, ResolvedType> vars,
                          Set<String> bare, JSONArray out) {
        String name = p.getNameAsString();
        bare.add(name);
        JSONObject o = new JSONObject().put("name", name);
        try {
            ResolvedType t = p.getType().resolve();
            vars.put(name, t);
            o.put("type", describe(t));
        } catch (Throwable ex) {
            o.put("type", p.getTypeAsString());
            o.put("unresolved", true);
            health.error("param " + name, ex);
        }
        out.put(o);
    }

    /**
     * Collect names available from the enclosing type at this program point:
     * {@code this}, accessible fields, and methods callable without qualification.
     *
     * Fields are recorded in the lexical scope but are not yet expanded as
     * receiver types. Receiver expansion is handled separately.
     */
    private void collectThisAndFields(TypeDeclaration<?> type, JSONObject receivers,
                                      Map<String, ResolvedType> vars, Set<String> bare,
                                      JSONObject scope, Config cfg) {
        JSONArray fields = new JSONArray();
        JSONArray statics = new JSONArray();
        if (type == null) {
            scope.put("fields", fields);
            scope.put("statics", statics);
            return;
        }
        try {
            ResolvedReferenceTypeDeclaration decl = type.resolve();

            JSONObject self = new JSONObject();
            self.put("type", decl.getQualifiedName());

            JSONObject methodInfo = methodsOfThis(decl, cfg);

            self.put("methods",
                methodInfo.getJSONArray("methods"));
            self.put("method_counts",
                methodInfo.getJSONObject("method_counts"));
            self.put("inherited_by_type",
                methodInfo.getJSONObject("inherited_by_type"));
            JSONArray fieldArr = new JSONArray();
            for (ResolvedFieldDeclaration f : safeVisibleFields(decl)) {
                fieldArr.put(new JSONObject()
                        .put("name", f.getName())
                        .put("static", f.isStatic()));
                bare.add(f.getName()); 
                fields.put(f.getName());
                if (f.isStatic()) statics.put(f.getName());
                try {
                    vars.putIfAbsent(f.getName(), f.getType());
                } catch (Throwable ignored) { /* field type may be unresolvable */ }
            }
            self.put("fields", fieldArr);
            receivers.put("this", self);

            // the enclosing type's own methods are callable bare
            JSONArray ms = self.getJSONArray("methods");
            for (int i = 0; i < ms.length(); i++) {
                bare.add(ms.getJSONObject(i).getString("name"));
            }
        } catch (Throwable ex) {
            health.error("enclosing type " + type.getNameAsString(), ex);
        }
        scope.put("fields", fields);
        scope.put("statics", statics);
    }

    private void collectImportsAndStatics(CompilationUnit cu, Set<String> bare, JSONObject scope, Map<String, String> importedTypes) {
        JSONArray types = new JSONArray();
        JSONArray staticImports = new JSONArray();
        boolean wildcard = false;
        for (ImportDeclaration imp : cu.getImports()) {
            if (imp.isAsterisk() && !imp.isStatic()) wildcard = true;
            String name = imp.getNameAsString();
            String simple = name.contains(".") ? name.substring(name.lastIndexOf('.') + 1) : name;
            if (imp.isStatic()) {
                if (!imp.isAsterisk()) {
                    staticImports.put(simple);
                    bare.add(simple);
                }
            } else if (!imp.isAsterisk()) {
                types.put(simple);
                bare.add(simple);
                importedTypes.put(simple, name);
            }
        }
        // types declared in this file, and siblings in the same directory
        cu.findAll(TypeDeclaration.class).forEach(t -> {
            types.put(t.getNameAsString());
            bare.add(t.getNameAsString());
        });
        cu.getStorage().ifPresent(st -> {
            File dir = st.getPath().getParent().toFile();
            File[] siblings = dir.listFiles((d, n) -> n.endsWith(".java"));
            if (siblings != null) {
                for (File f : siblings) {
                    String n = f.getName().substring(0, f.getName().length() - 5);
                    types.put(n);
                    bare.add(n);
                }
            }
        });
        scope.put("types", types);
        scope.put("static_imports", staticImports);
        // a wildcard import hides type names: the decoder must not constrain `new`
        scope.put("wildcard_imports", wildcard);
    }

    // ------------------------------------------------------------------
    // member enumeration -- the mask that matters
    // ------------------------------------------------------------------

    /** All members of a resolved type, inherited included, or null if unresolvable. */
    private JSONObject membersOf(ResolvedType t, Config cfg) {
        if (t == null || !t.isReferenceType()) return null;
        try {
            Optional<ResolvedReferenceTypeDeclaration> d = t.asReferenceType().getTypeDeclaration();
            if (d.isEmpty()) return null;
            ResolvedReferenceTypeDeclaration decl = d.get();
            JSONObject o = new JSONObject();
            o.put("type", decl.getQualifiedName());
            o.put("methods", methodsOf(decl, cfg));
            JSONArray fields = new JSONArray();
            for (ResolvedFieldDeclaration f : safeAllFields(decl)) {
                fields.put(new JSONObject().put("name", f.getName()).put("static", f.isStatic()));
            }
            o.put("fields", fields);
            if (decl.isEnum()) {
                JSONArray consts = new JSONArray();
                for (ResolvedEnumConstantDeclaration ec : decl.asEnum().getEnumConstants()) {
                    consts.put(ec.getName());
                }
                o.put("enum_constants", consts);
            }
            return o;
        } catch (Throwable ex) {
            health.error("members of " + safeDescribe(t), ex);
            return null;
        }
    }


        private static final String[] JAVA_LANG_TYPES = {
        "Math", "Integer", "Long", "Short", "Byte", "Double", "Float", "Boolean",
        "Character", "String", "System", "Thread", "Object", "Class", "Enum", "Number"
    };

    /**
     * Simple name -> fully qualified name of every type that can appear before a dot.
     * Java shadowing order, first wins: types in this file, single-type imports,
     * same-package types named in this file, then java.lang.
     */
    private Map<String, String> typeNameCandidates(CompilationUnit cu,
                                                   Map<String, String> importedTypes) {
        Map<String, String> out = new LinkedHashMap<>();
        // 1. types declared in this file (enclosing class, nested types)
        for (TypeDeclaration<?> t : cu.findAll(TypeDeclaration.class)) {
            t.getFullyQualifiedName().ifPresent(fq -> out.putIfAbsent(t.getNameAsString(), fq));
        }
        // 2. single-type imports
        importedTypes.forEach(out::putIfAbsent);
        // 3. same-package types that this file mentions
        Set<String> named = new HashSet<>();
        cu.findAll(SimpleName.class).forEach(n -> named.add(n.getIdentifier()));
        String pkg = cu.getPackageDeclaration().map(p -> p.getNameAsString() + ".").orElse("");
        cu.getStorage().ifPresent(st -> {
            File[] sib = st.getPath().getParent().toFile().listFiles((d, n) -> n.endsWith(".java"));
            if (sib == null) return;
            for (File f : sib) {
                String n = f.getName().substring(0, f.getName().length() - 5);
                if (named.contains(n)) out.putIfAbsent(n, pkg + n);
            }
        });
        // 4. java.lang (implicit import, lowest priority)
        for (String n : JAVA_LANG_TYPES) out.putIfAbsent(n, "java.lang." + n);
        return out;
    }

    private void collectUsedTypeNames(
            CallableDeclaration<?> method,
            int bugLine,
            JSONObject scope,
            Set<String> usedTypeNames) {

        JSONArray importedTypes = scope.optJSONArray("types");

        if (importedTypes == null) {
            return;
        }

        Set<String> allImported = new HashSet<>();

        for (int i = 0; i < importedTypes.length(); i++) {
            allImported.add(importedTypes.getString(i));
        }

        for (FieldAccessExpr fa : method.findAll(FieldAccessExpr.class)) {

            if (!fa.getRange().isPresent()) {
                continue;
            }

            if (fa.getRange().get().begin.line > bugLine) {
                continue;
            }

            Expression receiver = fa.getScope();

            if (receiver.isNameExpr()) {

                String name =
                        receiver.asNameExpr().getNameAsString();

                if (allImported.contains(name)) {
                    usedTypeNames.add(name);
                }
            }
        }

        for (MethodCallExpr call : method.findAll(MethodCallExpr.class)) {

            if (!call.getRange().isPresent()) {
                continue;
            }

            if (call.getRange().get().begin.line > bugLine) {
                continue;
            }

            call.getScope().ifPresent(scopeExpr -> {

                if (scopeExpr.isNameExpr()) {

                    String name =
                            scopeExpr.asNameExpr().getNameAsString();

                    if (allImported.contains(name)) {
                        usedTypeNames.add(name);
                    }
                }
            });
        }
    }

        private JSONObject staticMembersForTypeName(String qualifiedName,
                                                CombinedTypeSolver solver, Config cfg) {
        ResolvedReferenceTypeDeclaration decl = solveTypeName(qualifiedName, solver);
        return decl == null ? null : staticMembersOf(decl, cfg);
    }

    /** Resolves a type name; falls back to Outer + inner lookup for a.b.Outer.Inner. */
    private ResolvedReferenceTypeDeclaration solveTypeName(String fqn, CombinedTypeSolver solver) {
        try {
            var solved = solver.tryToSolveType(fqn);
            if (solved.isSolved()) return solved.getCorrespondingDeclaration();
        } catch (Throwable ignored) { /* try the nested form below */ }
        int dot = fqn.lastIndexOf('.');
        if (dot <= 0) return null;
        ResolvedReferenceTypeDeclaration outer = solveTypeName(fqn.substring(0, dot), solver);
        if (outer == null) return null;
        String inner = fqn.substring(dot + 1);
        try {
            for (ResolvedReferenceTypeDeclaration t : outer.internalTypes()) {
                if (t.getName().equals(inner)) return t;
            }
        } catch (Throwable ex) {
            health.error("member type " + fqn, ex);
        }
        return null;
    }

        /**
     * What may legally follow `Type.`: static methods, static fields (interface
     * constants are implicitly static), enum constants and member types.
     * All or nothing: if any part fails or hits the cap, return null so the
     * decoder leaves `Type.` unconstrained. A partial list would block legal code.
     */
    private JSONObject staticMembersOf(ResolvedReferenceTypeDeclaration decl, Config cfg) {
        try {
            JSONObject out = new JSONObject();
            out.put("type", decl.getQualifiedName());

            JSONArray staticMethods = new JSONArray();
            Set<String> seenMethods = new HashSet<>();
            for (MethodUsage m : decl.getAllMethods()) {
                if (!m.getDeclaration().isStatic()) continue;
                if (!seenMethods.add(m.getName() + "/" + m.getNoParams())) continue;
                JSONObject method = new JSONObject()
                        .put("name", m.getName())
                        .put("arity", m.getNoParams())
                        .put("static", true);
                try { method.put("returns", describe(m.returnType())); } catch (Throwable ignored) { }
                staticMethods.put(method);
            }

            JSONArray staticFields = new JSONArray();
            Set<String> seenFields = new HashSet<>();
            for (ResolvedFieldDeclaration f : decl.getAllFields()) {   // not safeAllFields
                boolean isStatic = f.isStatic() || f.declaringType().isInterface();
                if (isStatic && seenFields.add(f.getName())) {
                    staticFields.put(new JSONObject().put("name", f.getName()).put("static", true));
                }
            }

            if (decl.isEnum()) {
                JSONArray constants = new JSONArray();
                for (ResolvedEnumConstantDeclaration ec : decl.asEnum().getEnumConstants()) {
                    constants.put(ec.getName());
                }
                out.put("enum_constants", constants);
                // implicit enum statics, not always reported by the solver
                if (seenMethods.add("values/0")) staticMethods.put(new JSONObject()
                        .put("name", "values").put("arity", 0).put("static", true));
                if (seenMethods.add("valueOf/1")) staticMethods.put(new JSONObject()
                        .put("name", "valueOf").put("arity", 1).put("static", true));
            }

            // member types, own and inherited (JsonParser.NumberType, HashMap.Entry)
            JSONArray nested = new JSONArray();
            Set<String> seenTypes = new HashSet<>();
            for (ResolvedReferenceTypeDeclaration it : decl.internalTypes()) {
                if (seenTypes.add(it.getName())) nested.put(it.getName());
            }
            for (ResolvedReferenceType anc : decl.getAllAncestors()) {
                Optional<ResolvedReferenceTypeDeclaration> ad = anc.getTypeDeclaration();
                if (!ad.isPresent()) continue;
                for (ResolvedReferenceTypeDeclaration it : ad.get().internalTypes()) {
                    if (seenTypes.add(it.getName())) nested.put(it.getName());
                }
            }

            if (staticMethods.length() > cfg.maxMembersPerType
                    || staticFields.length() > cfg.maxMembersPerType) {
                health.error("static members of " + decl.getQualifiedName(),
                        new RuntimeException("over --max-members, left unconstrained"));
                return null;
            }

            out.put("static_methods", staticMethods);
            out.put("static_fields", staticFields);
            out.put("nested_types", nested);
            return out;
        } catch (Throwable ex) {
            String name;
            try { name = decl.getQualifiedName(); } catch (Throwable t) { name = "?"; }
            health.error("static members of " + name, ex);
            return null;
        }
    }
    /**
     * Method names with arity, inherited included. Arity is carried because a
     * later refinement can constrain the argument count too, not just the name.
     */
    private JSONArray methodsOf(ResolvedReferenceTypeDeclaration decl, Config cfg) {
        JSONArray arr = new JSONArray();
        Set<String> seen = new HashSet<>();
        try {
            for (MethodUsage m : decl.getAllMethods()) {
                if (arr.length() >= cfg.maxMembersPerType) break;
                String key = m.getName() + "/" + m.getNoParams();
                if (!seen.add(key)) continue;
                JSONObject o = new JSONObject()
                        .put("name", m.getName())
                        .put("arity", m.getNoParams());
                try {
                    o.put("returns", describe(m.returnType()));
                } catch (Throwable ignored) { /* generic returns can fail */ }
                arr.put(o);
            }
        } catch (Throwable ex) {
            health.error("methods of " + decl.getQualifiedName(), ex);
        }
        return arr;
    }

    /**
     * Methods legally callable through {@code this} from the current enclosing type.
     *
     * Private methods are callable only when they are declared by the current type.
     * Public and protected methods are callable.
     * Package-private methods are callable when the declaring type is in the same package.
     */
    private JSONObject methodsOfThis(ResolvedReferenceTypeDeclaration decl, Config cfg) {
        JSONArray all = new JSONArray();
        JSONArray declared = new JSONArray();
        JSONArray inherited = new JSONArray();
        Set<String> seen = new HashSet<>();
        Map<String, Integer> inheritedByType = new TreeMap<>();

        try {
            String currentPackage = decl.getPackageName();

            for (MethodUsage m : decl.getAllMethods()) {
                if (all.length() >= cfg.maxMembersPerType) break;

                ResolvedMethodDeclaration md = m.getDeclaration();
                String access = md.accessSpecifier().asString();

                ResolvedReferenceTypeDeclaration owner = m.declaringType();

                boolean sameType =
                        owner != null &&
                        owner.getQualifiedName().equals(decl.getQualifiedName());

                boolean samePackage =
                        owner != null &&
                        Objects.equals(owner.getPackageName(), currentPackage);

                // Private: only callable when declared in the current type.
                if ("private".equals(access) && !sameType) {
                    continue;
                }

                // Package-private: callable only from the same package.
                if (access.isEmpty() && !samePackage) {
                    continue;
                }

                // public and protected are accessible through `this`.
                String key = m.getName() + "/" + m.getNoParams();
                if (!seen.add(key)) continue;

                JSONObject o = new JSONObject()
                        .put("name", m.getName())
                        .put("arity", m.getNoParams());

                try {
                    o.put("returns", describe(m.returnType()));
                } catch (Throwable ignored) {
                    // Generic return types may occasionally fail to resolve.
                }

                all.put(o);
                // Split by declaring type.
                if (sameType) {
                    declared.put(o);
                } else {
                    inherited.put(o);
                    String ownerName = owner == null
                            ? "<unknown>"
                            : owner.getQualifiedName();

                    inheritedByType.merge(ownerName, 1, Integer::sum);
                }
            }
        } catch (Throwable ex) {
            health.error("visible methods of " + decl.getQualifiedName(), ex);
        }

        JSONObject result = new JSONObject();

        result.put("methods", all);

        result.put(
                "method_counts",
                new JSONObject()
                        .put("all", all.length())
                        .put("declared", declared.length())
                        .put("inherited", inherited.length()));

        JSONObject ownerCounts = new JSONObject();

        for (Map.Entry<String, Integer> e : inheritedByType.entrySet()) {
            ownerCounts.put(e.getKey(), e.getValue());
        }

        result.put("inherited_by_type", ownerCounts);
        // System.err.println("DEBUG inherited_by_type = " + ownerCounts);
        return result;
    }

    private List<ResolvedFieldDeclaration> safeAllFields(ResolvedReferenceTypeDeclaration decl) {
        try {
            return decl.getAllFields();
        } catch (Throwable ex) {
            health.error("fields of " + decl.getQualifiedName(), ex);
            return Collections.emptyList();
        }
    }

    private List<ResolvedFieldDeclaration> safeVisibleFields(
            ResolvedReferenceTypeDeclaration decl) {
        try {
            return decl.getVisibleFields();
        } catch (Throwable ex) {
            health.error("visible fields of " + decl.getQualifiedName(), ex);
            return Collections.emptyList();
        }
    }

    /**
     * Receivers that only ever appear as expressions -- {@code foo().bar},
     * {@code this.plot}, {@code a.b.c} -- which never show up in the local or
     * field list but are exactly the positions where the model invents names.
     */
    private void collectExpressionReceivers(CallableDeclaration<?> method,
                                            JSONObject receivers, Config cfg) {
        for (MethodCallExpr call : method.findAll(MethodCallExpr.class)) {
            call.getScope().ifPresent(sc -> addReceiverFor(sc, receivers, cfg));
        }
        for (FieldAccessExpr fa : method.findAll(FieldAccessExpr.class)) {
            addReceiverFor(fa.getScope(), receivers, cfg);
        }
    }

    private void addReceiverFor(Expression scope, JSONObject receivers, Config cfg) {
        String key = scope.toString();
        if (key.length() > 120 || receivers.has(key)) return;
        health.receiversAttempted++;
        try {
            JSONObject members = membersOf(scope.calculateResolvedType(), cfg);
            if (members != null) {
                health.receiversResolved++;
                receivers.put(key, members);
            }
        } catch (Throwable ex) {
            health.error("receiver " + trim(key, 60), ex);
        }
    }

    // ------------------------------------------------------------------
    // enums
    // ------------------------------------------------------------------

    /**
     * The switch that encloses the hole, if its selector is an enum. Only then
     * may the decoder restrict {@code case} labels; an int/String switch stays
     * unconstrained.
     */
    private void collectHoleSwitch(Node hole, JSONObject root) {
        Expression selector = null;
        SwitchStmt ss = enclosing(hole, SwitchStmt.class);
        if (ss != null) selector = ss.getSelector();
        else {
            SwitchExpr se = enclosing(hole, SwitchExpr.class);
            if (se != null) selector = se.getSelector();
        }
        if (selector == null) return;
        try {
            ResolvedType t = selector.calculateResolvedType();
            if (!t.isReferenceType()) return;
            Optional<ResolvedReferenceTypeDeclaration> d = t.asReferenceType().getTypeDeclaration();
            if (d.isEmpty() || !d.get().isEnum()) return;
            JSONArray consts = new JSONArray();
            for (ResolvedEnumConstantDeclaration ec : d.get().asEnum().getEnumConstants()) {
                consts.put(ec.getName());
            }
            root.put("hole_switch_enum", d.get().getQualifiedName());
            root.put("hole_switch_constants", consts);
        } catch (Throwable ex) {
            health.error("hole switch selector", ex);
        }
    }
 

    private void collectSwitchEnums(CallableDeclaration<?> method, JSONObject enums, Set<String> bare) {
        for (SwitchStmt sw : method.findAll(SwitchStmt.class)) {
            try {
                addEnumConstants(sw.getSelector().calculateResolvedType(), enums, bare);
            } catch (Throwable ex) {
                health.error("switch selector", ex);
            }
        }
        for (SwitchExpr sw : method.findAll(SwitchExpr.class)) {
            try {
                addEnumConstants(sw.getSelector().calculateResolvedType(), enums, bare);
            } catch (Throwable ex) {
                health.error("switch expr selector", ex);
            }
        }
    }

    private void addEnumConstants(ResolvedType t, JSONObject enums, Set<String> bare) {
        if (t == null || !t.isReferenceType()) return;
        try {
            Optional<ResolvedReferenceTypeDeclaration> d = t.asReferenceType().getTypeDeclaration();
            if (d.isEmpty() || !d.get().isEnum()) return;
            ResolvedReferenceTypeDeclaration decl = d.get();
            if (enums.has(decl.getQualifiedName())) return;
            JSONArray consts = new JSONArray();
            for (ResolvedEnumConstantDeclaration ec : decl.asEnum().getEnumConstants()) {
                consts.put(ec.getName());
                bare.add(ec.getName());
            }
            enums.put(decl.getQualifiedName(), consts);
        } catch (Throwable ex) {
            health.error("enum constants", ex);
        }
    }

    // ------------------------------------------------------------------
    // helpers
    // ------------------------------------------------------------------

    static String describe(ResolvedType t) {
        try {
            if (t.isReferenceType()) return t.asReferenceType().getQualifiedName();
            if (t.isPrimitive()) return ((ResolvedPrimitiveType) t).describe();
            return t.describe();
        } catch (Throwable ex) {
            return "?";
        }
    }

    private String safeDescribe(ResolvedType t) {
        try {
            return t.describe();
        } catch (Throwable ex) {
            return "?";
        }
    }
}

// // /u/student/2023/cs23resch16001/call-graph-copy/defects4j_checkouts/Chart_1b/source/org/jfree/chart/renderer/category/AbstractCategoryItemRenderer.java

// java -cp "target/classes:deps.jar" com.example.GroundedSymbolTable \
//   --file /u/student/2023/cs23resch16001/call-graph-copy/defects4j_checkouts/Chart_1b/source/org/jfree/chart/renderer/category/AbstractCategoryItemRenderer.java \
//   --line 1564 \
//   --source-root /u/student/2023/cs23resch16001/call-graph-copy/defects4j_checkouts/Chart_1b/source \
//   --classpath "$(defects4j export -p cp.compile -w /u/student/2023/cs23resch16001/call-graph-copy/defects4j_checkouts/Chart_1b)" \
//   --out symbols/Chart_1b_getLegendItems.json \
//   --pretty
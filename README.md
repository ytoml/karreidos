# Karreidos

[Kaleidoscope](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/index.html) in Rust but somewhat modified.

Main differences from original Kaleidoscope:
- Function definition with `fn` instead of `def`
- Comment with `//` or `/* */`, not with `#`
- Loop statement with genarator-like syntax (`for i <- 1..10, 1 { /* .. */ }`)
- Pipe operator `/>`

You can open REPL with:
```
cargo build --release
./target/release/karreidos -i
```
then, you can write some codes like:
```karreidos
?>> extern fn cos(x); // from standard C library
?>> fn add(x, y) { x + y; }
?>> fn double(x) { x * 2; }
?>> fn func(arg) {
...     if arg < 10 {
...         let mut i = 10;
...         for mut v <- arg .. 100, 1 {
...             i = i + v;
...             v = v + 1; // number of repetition is not affected by this assignment
...         };
...         i /> add(3);
...     } else {
...         cos(arg);
...     }
...         /> double();
... }
...
?>> func(9);
[INFO] 9854  // (10 + sum(9..=99) + 3) * 2
?>> func(31.415926535);
[INFO] 2     // cos(10pi) * 2
```
Grammer is available <a href='./grammer.md'>here</a>.
Note that redefining variable in the same scope (block) is prohibited.
```karreidos
?>> fn func() {
...     let i = 10;
...     let i = 10;
... }
...
[ERROR] Redefined("i", Variable)
```

## To be implemented
- [ ] Kaleidoscope itself.
    current progress: chapter 7
- [ ] Language server protocol
- [ ] Port them using tree-sitter

# Karreidos

[Kaleidoscope](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/index.html) in Rust but somewhat modified.

Main differences from original Kaleidoscope:
- function definition with `fn` instead of `def`
- comment with `//` or `/* */`, not with `#`

You can open REPL with:
```
cargo build --release
./target/release/karreidos -i
```
then, you can write some codes like:
```karreidos
?> fn func(x) {
       if x {
            x + 1;
       } else {
            0;
       };
   }
?> func(1);
[INFO] 2
?> func(0);
[INFO] 0
```
Grammer is available <a href='./grammer.md'>here</a>.

## To be implemented
- [ ] Kaleidoscope itself.
    current progress: chapter 3
- [ ] Language server protocol
- [ ] Port them using tree-sitter

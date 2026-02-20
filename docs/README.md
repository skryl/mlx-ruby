## Build the Docs

### Setup (do once)

Install Doxygen:

```
brew install doxygen
```

Install Ruby packages:

```
bundle install
```

### Build

Build the docs from `mlx/docs/`

```
doxygen && make html
```

View the docs by running a server in `mlx/docs/build/html/`:

```sh
ruby -run -e httpd mlx/docs/build/html -p <port>
```

and point your browser to `http://localhost:<port>`.

### Push to GitHub Pages

Check-out the `gh-pages` branch (`git switch gh-pages`) and build
the docs. Then force add the `build/html` directory:

`git add -f build/html`

Commit and push the changes to the `gh-pages` branch.

## Doc Development Setup

To enable live refresh of docs while writing:

Install sphinx autobuild
```
bundle install
```

Run auto build on docs/src folder
```
bundle exec rake docs:watch
```

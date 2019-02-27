# How to contribute

There are many possible variants of distributed dictionary learning, and we hope that
this code provides a starting place for variations both large and small.

Please do pull the code, run the code, and create a topic branch for your modifications.
Typically, at the outset of this project, you will create a topic branch from the master branch, but, as this
project evolves, we envisage various branches with significant differences.

We will review your changes to either incorporate into the master branch, into a different branch,
or keep as a separate new branch.

Fork, then clone the repo:

    git clone git@github.com:your-username/distributed-dictionary-learning.git

Set up your machine:

    ./bin/setup

Make sure the tests pass:

    rake

Make your change. Add tests for your change. Make the tests pass:

    rake

Push to your fork and submit a pull request against the appropriate branch.
For example, [submit a master pull request][pr].

[pr]: https://github.com/att/distributed-dictionary-learning/compare/

At this point you're waiting on us. We like to at least comment on pull requests
within five business days (and, typically, less that that). We may suggest
some changes or improvements or alternatives.

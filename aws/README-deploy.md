
## Always `sam build` before `sam deploy`

`sam deploy` ships whatever `sam build` last produced. It does NOT rebuild, and
it does NOT warn you that the image predates your source.

This bit once, silently: a fix was written 82 seconds after a build, then
deployed. CloudFormation reported UPDATE_COMPLETE, every Lambda showed a fresh
LastModified, and the stack was running month-old logic for that one function.
Nothing in the deploy output indicated a problem.

The only reliable check is timestamps:

    [ execution.py -nt aws/.aws-sam/build.toml ] && echo "STALE BUILD"

or just always run both:

    cd aws && sam build --profile alpaca && \
              sam deploy --config-env dev  --profile alpaca --no-confirm-changeset && \
              sam deploy --config-env prod --profile alpaca --no-confirm-changeset

Note the config-env names are `dev` and `prod` -- there is no `default`, and
`--config-env default` fails with a confusing "Missing option '--stack-name'".

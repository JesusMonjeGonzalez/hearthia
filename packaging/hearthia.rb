class Hearthia < Formula
  desc "The self-tending fire for local models — control plane for llama.cpp on Apple Silicon"
  homepage "https://github.com/JesusMonjeGonzalez/hearthia"
  url "https://github.com/JesusMonjeGonzalez/hearthia.git",
      tag:      "v0.2.0",
      revision: "0000000000000000000000000000000000000000"
  license "MIT"

  depends_on arch: :arm64
  depends_on :macos
  depends_on "python@3.12"

  def install
    venv = virtualenv_create(libexec, "python3.12")
    venv.pip_install_and_link buildpath
  end

  def caveats
    <<~EOS
      Hearthia manages llama.cpp and llama-swap; install them first:

          brew install llama.cpp llama-swap

      Then create the stack config and install the services:

          mkdir -p ~/.hearthia/models
          $EDITOR ~/.hearthia/llama-swap.yaml
          hearth install && hearth doctor

      Try the product without any setup first:

          hearth demo
    EOS
  end

  test do
    assert_match "Hearthia", shell_output("#{bin}/hearth version")
  end
end

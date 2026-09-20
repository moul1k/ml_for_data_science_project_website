(() => {
  const pages = [...document.querySelectorAll('[data-page]')];
  const links = [...document.querySelectorAll('[data-go]')];
  const names = pages.map(page => page.id);
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  document.documentElement.classList.add('js');

  function navigate() {
    const name = decodeURIComponent(location.hash.slice(1));
    const open = names.includes(name);
    document.body.classList.toggle('viewing', open);
    pages.forEach(page => {
      page.hidden = !open || page.id !== name;
      page.classList.toggle('visible', open && page.id === name);
    });
    links.forEach(link => {
      const active = open && link.dataset.go === name;
      link.classList.toggle('active', active);
      if (active) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    });
    if (open) window.scrollTo({top: 0, behavior: reduced.matches ? 'instant' : 'smooth'});
  }

  links.forEach(link => link.addEventListener('click', () => {
    if (location.hash === '#' + link.dataset.go) navigate();
    else location.hash = link.dataset.go;
  }));
  document.getElementById('overview').addEventListener('click', () => {
    history.pushState(null, '', location.pathname + location.search);
    navigate();
    window.scrollTo({top: 0, behavior: reduced.matches ? 'instant' : 'smooth'});
  });
  document.getElementById('home-link').addEventListener('click', event => {
    event.preventDefault();
    history.pushState(null, '', location.pathname + location.search);
    navigate();
    window.scrollTo({top: 0, behavior: reduced.matches ? 'instant' : 'smooth'});
  });
  window.addEventListener('hashchange', navigate);
  window.addEventListener('popstate', navigate);
  navigate();
})();
